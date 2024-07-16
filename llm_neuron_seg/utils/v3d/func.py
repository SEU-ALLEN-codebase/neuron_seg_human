import os
import subprocess
import logging
import sys

"""
    sort_swc(swc_in, swc_out): 对 SWC 文件进行排序。
    radius_swc(img, swc, swc_out): 计算神经元的半径。
    swc2img(swc_in, img_out, p): 将 SWC 文件转换为图像。
    app2:tracing"""


class Vaa3DPlugin:

    def __init__(self, v3d_path=None):
        self.v3d_path = v3d_path or self._default_v3d_path()
        logging.basicConfig(filename='vaa3d_plugins.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    def _default_v3d_path(self):
        # 路径根据操作系统j进行分离设置
        if sys.platform.startswith('linux'):
            #添加linux v3d 路径
            return "/usr/local/Vaa3D/Vaa3D"
        elif sys.platform.startswith('win'):
            return r"C:\Users\SEU\Desktop\Vaa3D-x.1.1.4_Windows_64bit_version\Vaa3D-x.1.1.4_Windows_64bit_version\Vaa3D-x.exe"
        else:
            raise EnvironmentError("Unsupported platform")

    def run_command(self, cmd_args):
        if sys.platform.startswith('linux'):
            cmd_args = ['xvfb-run', '-a','-s', '"-screen 0 640x480x16"'] + cmd_args

        process = subprocess.Popen(cmd_args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode()
            print(f"Error: {error_message}")
            logging.error(f"allen!! Failed command: {' '.join(cmd_args)}\nError: {error_message}")
        else:
            success_message = stdout.decode()
            #logging.info(f"Success command: {' '.join(cmd_args)}\nOutput: {success_message}")

    def sort_swc(self, swc_in, swc_out=None):
        if sys.platform.startswith('linux'):
            cmd_args = [self.v3d_path, '-x', 'sort_neuron_swc', '-f', 'sort_swc', '-i', swc_in, '-o', swc_out]
        else:
            cmd_args = [self.v3d_path, "/x", "sort_neuron_swc", "/f", "sort_swc", "/i", swc_in, "/o", swc_out]
        self.run_command(cmd_args)

    def radius_swc(self, img, swc, swc_out):
        if sys.platform.startswith('linux'):
            cmd_args = [self.v3d_path, '-x', 'neuron_radius', '-f', 'neuron_radius', '-i', img, swc, '-o', swc_out]
        else:
            cmd_args = [self.v3d_path, "/x", "neuron_radius", "/f", "neuron_radius", "/i", img, swc, "/o", swc_out]
        self.run_command(cmd_args)

    def swc2img(self, swc_in, img_out, p):
        #p={image.shape[2]}{image.shape[1]}{image.shape[0]}
        if sys.platform.startswith('linux'):
    #
            cmd_args = [self.v3d_path, '-x', 'swc_to_maskimage_sphere_unit', '-f', 'swc_to_maskimage', '-i', swc_in, '-o', swc_path, '-p', p,'-o', img_out]
        else:
            cmd_args = [self.v3d_path, "/x", "swc_to_maskimage_sphere_unit", "/f", "swc_to_maskimage", "/i", swc_in, "/p", p, "/o", img_out]
        self.run_command(cmd_args)

    def neuron_tracing(self, skelwithsoma_path, swc_path, somamarker_path):
        resample = 1
        gsdt = 0
        b_RadiusFrom2D = 1

        if not os.path.exists(somamarker_path):
            somamarker_path = "NULL"

        if sys.platform.startswith('linux'):
            cmd_args = [self.v3d_path, '-x', 'vn2', '-f', 'app1', '-i', skelwithsoma_path, '-o', swc_path, '-p', somamarker_path, '0', 'AUTO', '1', str(b_RadiusFrom2D), str(gsdt), '1', '5', str(resample), '0', '0']
        else:
                #vn2_path = 'D:/tracing_ws/vn2.dll'
            cmd_args = [self.v3d_path, "/x", "vn2", "/f", "app1", "/i", skelwithsoma_path, "/o", swc_path, "/p", somamarker_path, "0", "10", "1"]
        self.run_command(cmd_args)


def main():
   
    v3d_test = Vaa3DPlugin()
    
    input_swc = "input.swc"
    output_sorted_swc = "output_sorted.swc"
    input_img = "image.tif"
    output_radius_swc = "output_radius.swc"
    output_mask_img = "output_image.tif"

    #print("Sorting SWC file...")
    v3d_test.sort_swc(input_swc, output_sorted_swc) 
    #print("Calculating neuron radius...")
    v3d_test.radius_swc(input_img, output_sorted_swc, output_radius_swc)
    #print("Converting SWC to image...")
    #v3d_test.swc2img(output_radius_swc, output_mask_img, dimensions)

if __name__ == "__main__":
    main()
