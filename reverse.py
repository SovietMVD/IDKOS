import os
from PIL import Image, ImageOps

# 指定图片所在的文件夹路径和处理后图片保存的路径
input_folder = 'dataset/text feature/bind_Train2.0'
output_folder = 'dataset/text feature/bind_Train2.0_reverse'

# 确保输出文件夹存在


# 遍历文件夹中的所有文件
for P in os.listdir(input_folder):
    dir_path = os.path.join(input_folder, P)
    out_path = os.path.join(output_folder, P)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for filename in os.listdir(dir_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 只处理图片格式的文件
            img_path = os.path.join(dir_path, filename)
            out_img_path = os.path.join(out_path, filename)

        # 打开图片
        with Image.open(img_path) as img:
            # 反色处理
                inverted_image = ImageOps.invert(img.convert("RGB"))


            # 保存反色后的图片
                output_path = out_img_path
                inverted_image.save(output_path)

                print(f"已处理图片: {output_path}")
