
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Written by @dlazares

"""Processes a synced multi-camera video dataset to a nerfstudio compatible dataset."""

import json
import numpy as np
import shutil
import torch
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
from pathlib import Path

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.process_data import colmap_utils
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)


@dataclass
class MulticamVideoToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process multiple time synced camera videos into a nerfstudio dataset.

    This script does the following:

    1. Converts each camera's video into images and downscales them. 
    2. Calculates the camera poses for the initial timestep using `COLMAP <https://colmap.github.io/>`_.
    """

    num_frames_target: int = 300
    """Target number of frames to use per video, results may not be exact."""
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "sequential"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed
    and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
    should only be used for videos."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""

        # Get mp4s in self.data path
        videos = [f for f in self.data.glob("*.mp4")]
        num_cams = len(videos)
        # convert the videos into images stored in per camera folders

        summary_log = []
        summary_log_eval = []
        # Convert video to images
        num_extracted_frames = 0
        print(f"processing {self.camera_type}")
        for i, video in enumerate(videos):
            cam_name = video.stem
            per_cam_dir = self.output_dir / f"{cam_name}"
            per_cam_images_dir = per_cam_dir / "images"
            if self.camera_type == "equirectangular":
                # create temp images folder to store the equirect and perspective images
                temp_image_dir = per_cam_dir / "temp_images"
                temp_image_dir.mkdir(parents=True, exist_ok=True)
                summary_log, num_extracted_frames_cam = process_data_utils.convert_video_to_images(
                    video,
                    image_dir=temp_image_dir,
                    num_frames_target=self.num_frames_target,
                    num_downscales=0,
                    crop_factor=(0.0, 0.0, 0.0, 0.0),
                    verbose=self.verbose,
                )

                if num_extracted_frames_cam != self.num_frames_target:
                    print(f"extracted {num_extracted_frames_cam} but target was {self.num_frames_target}")
                num_extracted_frames = num_extracted_frames_cam


                # Generate planar projections if equirectangular
                if self.eval_data is not None:
                    raise ValueError("Cannot use eval_data with camera_type equirectangular.")

                perspective_image_size = equirect_utils.compute_resolution_from_equirect(
                    temp_image_dir, self.images_per_equirect
                )

                equirect_utils.generate_planar_projections_from_equirectangular(
                    temp_image_dir,
                    perspective_image_size,
                    self.images_per_equirect,
                    crop_factor=self.crop_factor,
                )

                # copy the perspective images to the image directory
                process_data_utils.copy_images(
                    temp_image_dir / "planar_projections",
                    image_dir=per_cam_dir / "images",
                    verbose=False,
                )

                # remove the temp_images folder
                shutil.rmtree(self.output_dir / "temp_images", ignore_errors=True)

                self.camera_type = "perspective"

                # # Downscale images
                summary_log.append(
                    process_data_utils.downscale_images(per_cam_dir, self.num_downscales, verbose=self.verbose)
                )

            else:
                # If we're not dealing with equirects we can downscale in one step.
                summary_log, num_extracted_frames_cam = process_data_utils.convert_video_to_images(
                    video,
                    image_dir=per_cam_images_dir,
                    num_frames_target=self.num_frames_target,
                    num_downscales=self.num_downscales,
                    crop_factor=self.crop_factor,
                    verbose=self.verbose,
                    image_prefix="frame_train_" if self.eval_data is not None else "frame_",
                    keep_image_dir=False,
                )
                if num_extracted_frames_cam != self.num_frames_target:
                    print(f"extracted {num_extracted_frames_cam} but target was {self.num_frames_target}")
                num_extracted_frames = num_extracted_frames_cam

                if self.eval_data is not None:
                    summary_log_eval, num_extracted_frames_eval = process_data_utils.convert_video_to_images(
                        self.eval_data,
                        image_dir=per_cam_images_dir,
                        num_frames_target=self.num_frames_target,
                        num_downscales=self.num_downscales,
                        crop_factor=self.crop_factor,
                        verbose=self.verbose,
                        image_prefix="frame_eval_",
                        keep_image_dir=True,
                    )
                    summary_log += summary_log_eval
                    num_extracted_frames += num_extracted_frames_eval

            # Create mask
            # mask_path = process_data_utils.save_mask(
            #     image_dir=per_cam_images_dir,
            #     num_downscales=self.num_downscales,
            #     crop_factor=(0.0, 0.0, 0.0, 0.0),
            #     percent_radius=self.percent_radius_crop,
            # )
            # if mask_path is not None:
            #     summary_log.append(f"Saved mask to {mask_path}")


            # Export depth maps
            # image_id_to_depth_path, log_tmp = self._export_depth()
            # summary_log += log_tmp

            # Copy frames at target timestep t for initial frame colmap
            first_frame_path = per_cam_images_dir / "frame_00001.png"
            assert(first_frame_path.exists() == True)
            shutil.copy(first_frame_path,self.image_dir / f"{video.stem}.png")

        assert(num_extracted_frames > 0)

        # Run Colmap
        if not self.skip_colmap:
            # TODO: consider adding back mask path
            self._run_colmap(None)
        summary_log += self._save_transforms(num_extracted_frames, [f.stem for f in videos], None,None)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)

    def _save_transforms(
        self,
        num_frames: int,
        cam_names: List[str],
        image_id_to_depth_path: Optional[Dict[int, Path]] = None,
        camera_mask_path: Optional[Path] = None,
        image_rename_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Save colmap transforms into the output folder

        Args:
            image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
            image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        """
        summary_log = []
        frames = []
        if (self.absolute_colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                # Load Colmap images and cameras
                recon_dir = self.absolute_colmap_model_path
                cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
                im_id_to_image = read_images_binary(recon_dir / "images.bin")
                keep_original_world_coordinate = False
                ply_filename = "sparse_pc.ply"
                for im_id, im_data in im_id_to_image.items():
                        # NB: COLMAP uses Eigen / scalar-first quaternions
                        # * https://colmap.github.io/format.html
                        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
                        # the `rotation_matrix()` handles that format for us.

                        # TODO(1480) BEGIN use pycolmap API
                        # rotation = im_data.rotation_matrix()
                        rotation = qvec2rotmat(im_data.qvec)

                        translation = im_data.tvec.reshape(3, 1)
                        w2c = np.concatenate([rotation, translation], 1)
                        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
                        c2w = np.linalg.inv(w2c)
                        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
                        c2w[0:3, 1:3] *= -1
                        if not keep_original_world_coordinate:
                            c2w = c2w[np.array([0, 2, 1, 3]), :]
                            c2w[2, :] *= -1

                        # TODO: Match video camera name to camera id
                        name = im_data.name
                        if image_rename_map is not None:
                            name = image_rename_map[name]

                        # Write all frames per camera
                        # index from 1 for frames?? why nerfstudio?! why?!
                        for frame_num in range(1,num_frames+1):
                            cam_name = Path(im_data.name).stem
                            nice_frame_num = "{:05d}".format(frame_num)

                            name = Path(f"./{cam_name}/images/{nice_frame_num}.png")

                            frame = {
                                "file_path": name.as_posix(),
                                "transform_matrix": c2w.tolist(),
                                "colmap_im_id": im_id,
                                "time": (frame_num - 1 / num_frames)
                            }
                            # TODO add back camera mask and depth path
                            #if camera_mask_path is not None:
                            #    frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
                            #if image_id_to_depth_path is not None:
                            #    depth_path = image_id_to_depth_path[im_id]
                            #    frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))
                            frames.append(frame)

                if set(cam_id_to_camera.keys()) != {1}:
                    raise RuntimeError("Only single camera shared for all images is supported.")
                out = colmap_utils.parse_colmap_camera_params(cam_id_to_camera[1])
                out["frames"] = frames

                applied_transform = None
                if not keep_original_world_coordinate:
                    applied_transform = np.eye(4)[:3, :]
                    applied_transform = applied_transform[np.array([0, 2, 1]), :]
                    applied_transform[2, :] *= -1
                    out["applied_transform"] = applied_transform.tolist()

                # create ply from colmap
                assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
                colmap_utils.create_ply_from_colmap(
                    ply_filename,
                    recon_dir,
                    self.output_dir,
                    torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
                )
                out["ply_file_path"] = ply_filename

                with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=4)

                num_cams = len(cam_names)
                num_matched_frames = len(im_id_to_image) 
                summary_log.append(f"Colmap matched {num_matched_frames} images")
                summary_log.append(colmap_utils.get_matching_summary(num_cams, num_matched_frames))

        else:
            CONSOLE.log(
                "[bold yellow]Warning: Could not find existing COLMAP results. " "Not generating transforms.json"
            )
        return summary_log

