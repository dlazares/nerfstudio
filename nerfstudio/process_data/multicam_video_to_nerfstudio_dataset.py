
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

import os
import json
import numpy as np
import shutil
import torch
import subprocess
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import resize
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
from pathlib import Path
from PIL import Image


from nerfstudio.utils.misc import torch_compile
from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.process_data import colmap_utils
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)

def image_path_to_tensor(image_path: Path, size: Optional[tuple] = None, black_and_white=False) -> Tensor:
    """Convert an image from path to a tensor."""
    img = Image.open(image_path).convert("1") if black_and_white else Image.open(image_path)
    img_tensor = transforms.ToTensor()(img).permute(1, 2, 0)[..., :3]
    if size:
        img_tensor = resize(img_tensor.permute(2, 0, 1), size=size).permute(1, 2, 0)
    return img_tensor

def save_depth_tensor_to_png(tensor, file_path):
    """
    Save a depth tensor to a PNG file.

    Parameters:
        tensor (torch.Tensor): A 2D depth tensor.
        file_path (str): Path to save the PNG file.
    """
    # Ensure tensor is in CPU and detach it from the computation graph
    tensor = tensor.detach().cpu()

    # Normalize the tensor to 0-255 and convert to uint8
    normalized_tensor = 255 * (tensor - tensor.min()) / (tensor.max() - tensor.min())
    image = normalized_tensor.to(torch.uint8)

    # Convert to PIL Image and save
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(file_path, 'PNG')


@dataclass
class MulticamVideoToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process multiple time synced camera videos into a nerfstudio dataset.

    This script does the following:

    1. Converts each camera's video into images and downscales them. 
    2. Calculates the camera poses for the initial timestep using `COLMAP <https://colmap.github.io/>`_.
    """

    skip_extract:bool = False 
    use_depth:bool = True 
    skip_depth_extract:bool = False 
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
        print(f"NUM CAMS: {num_cams}")
        # convert the videos into images stored in per camera folders

        summary_log = []
        summary_log_eval = []
        # Convert video to images
        num_extracted_frames = 0
        print(f"processing {self.camera_type}")
        if self.skip_extract:
            print(f"\n Skipping video extract \n")
        if not self.skip_extract:
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

        if num_extracted_frames <= 0:
            print(" WARNING: num frames extracted was 0")

        # Run Colmap
        cam_names = [f.stem for f in videos]
        if not self.skip_colmap:
            # TODO: consider adding back mask path
            self._run_colmap(None)
            # Undistort multi-cam frames
            self._undistort(cam_names)

        # Extract per-frame Depth Maps
        if not self.skip_depth_extract:
            self._export_depth(multicam_names=cam_names)

        if self.skip_extract and self.skip_colmap:
            num_extracted_frames = self.num_frames_target

        summary_log += self._save_transforms(num_extracted_frames, cam_names, None,None)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)


    def _undistort(self, cam_names: List[str]):
        distorted_colmap_model_path = self.output_dir / Path("colmap/distorted/sparse/0")
        

        if (distorted_colmap_model_path / "cameras.bin").exists():
            # Load Distorted Colmap images and cameras
            recon_dir = distorted_colmap_model_path
            cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = read_images_binary(recon_dir / "images.bin")
            cam_name_to_cam_id = {}

            for cam_name in cam_names:
                for im_id, im_data in im_id_to_image.items():
                    if cam_name in im_data.name:
                        cam_name_to_cam_id[cam_name] = im_data.camera_id

            for cam_name in cam_names:
                per_cam_dir = self.output_dir / f"{cam_name}"
                per_cam_images_dir = per_cam_dir / "images"
                output_filename = self.output_dir / f"undistort_{cam_name}.txt"
                cam_id = cam_name_to_cam_id[cam_name]
                cam = cam_id_to_camera[cam_id]

                to_write = [cam.model, cam.width, cam.height, *cam.params]
                cam_params = " ".join([str(elem) for elem in to_write])
                distorted_images = sorted(os.listdir(per_cam_images_dir))

                with open(output_filename, 'w') as file:
                    for image_path in distorted_images:
                       file.write(f"{image_path} {cam_params}\n")
                    print(f"\nINFO: File '{output_filename}' written successfully with camera parameters.")

      # Run Standalone Image Undistorter
                output_path = (self.output_dir / f"colmap/undistorted/{cam_name}/images")
                output_path.mkdir(exist_ok=True,parents=True)

                undistort_command = f"colmap image_undistorter_standalone --input_file {output_filename} --output_path {output_path} --image_path {per_cam_images_dir}"
      
                subprocess.run(undistort_command, shell=True, check=True)
                process_data_utils.downscale_images(
                    output_path,
                    self.num_downscales,
                    folder_name="images",
                    nearest_neighbor=True,
                    verbose=self.verbose,
                )

    def _export_depth(self,multicam_names=[]) -> Tuple[Optional[Dict[int, Path]], List[str]]:
        """This method will create the depth images for multi-cam directories.

        Returns:
            Depth file paths indexed by COLMAP image id, logs
        """
        summary_log = []
        if self.use_depth and not self.skip_depth_extract:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            repo = "isl-org/ZoeDepth"
            depth_model = torch_compile(torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device))
            for cam_name in multicam_names:
                per_cam_dir = self.output_dir / f"colmap/undistorted/{cam_name}"
                print(f"Processing depth for {cam_name}. writing to {per_cam_dir}")
                per_cam_images_dir = per_cam_dir / "images"
                per_cam_depth_dir = per_cam_dir / "depths"
                per_cam_depth_dir.mkdir(exist_ok=True,parents=True)
                cam_images = per_cam_images_dir.glob("*.png")  # TODO: support jpg
                # TODO:  Downsample size for directories images_2, images_4, and images_8
                with torch.no_grad():
                    for image_path in cam_images:
                        img_tensor = image_path_to_tensor(image_path)
                        image = torch.permute(img_tensor, (2, 0, 1)).unsqueeze(0).to(device)
                        depth_image = depth_model.infer(image).squeeze()#.unsqueeze(-1)
                        depth_path =  per_cam_depth_dir / (str(Path(image_path).stem) + str(Path(image_path.suffix)))
                        save_depth_tensor_to_png(depth_image,depth_path)
                summary_log.append(
                    process_data_utils.downscale_images(
                        per_cam_images_dir,
                        self.num_downscales,
                        folder_name="depths",
                        nearest_neighbor=True,
                        verbose=self.verbose,
                    )
                )
        return None, summary_log

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
        out = {}
        if (self.absolute_colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                # Load Colmap images and cameras
                recon_dir = self.absolute_colmap_model_path
                cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
                im_id_to_image = read_images_binary(recon_dir / "images.bin")
                keep_original_world_coordinate = False
                ply_filename = "sparse_pc.ply"
                print("cam keys",len(cam_id_to_camera.keys()),cam_id_to_camera.keys())
                print("image keys",len(im_id_to_image.keys()))
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


                        per_cam_out = colmap_utils.parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id])

                        # Write all frames per camera
                        # index from 1 for frames?? why nerfstudio?! why?!
                        for frame_num in range(1,num_frames+1):
                            cam_name = Path(im_data.name).stem
                            nice_frame_num = "{:05d}".format(frame_num)

                            name = Path(f"./colmap/undistorted/{cam_name}/images/frame_{nice_frame_num}.png")
                            depth_name = Path(f"./colmap/undistorted/{cam_name}/depths/frame_{nice_frame_num}.png")

                            frame = {
                                "file_path": name.as_posix(),
                                "transform_matrix": c2w.tolist(),
                                "colmap_im_id": im_id,
                                "time": (frame_num - 1) / num_frames
                            }
                            if self.use_depth:
                                if (self.output_dir / depth_name).exists():
                                    frame["depth_file_path"] = depth_name.as_posix()
                                else:
                                    print(f"missing depth path {depth_name} at {self.output_dir}")
                            frame.update(per_cam_out)
                            # TODO add back camera mask and depth path
                            #if camera_mask_path is not None:
                            #    frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
                            #if image_id_to_depth_path is not None:
                            #    depth_path = image_id_to_depth_path[im_id]
                            #    frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))
                            frames.append(frame)

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

