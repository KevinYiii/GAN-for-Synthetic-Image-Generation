import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import Canvas, Scrollbar
import torch
import torch.nn as nn
from tkinter import Button


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 768, 4, 1, 0, bias=False),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.15, inplace=True),

            nn.ConvTranspose2d(768, 384, 4, 2, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.15, inplace=True),

            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.15, inplace=True),
            ResidualBlock(192),

            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.15, inplace=True),

            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.15, inplace=True),

            nn.ConvTranspose2d(48, 24, 4, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.15, inplace=True),

            nn.ConvTranspose2d(24, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z.view(-1, latent_size, 1, 1))


latent_size = 175

device = torch.device('cpu')

G = Generator(latent_size)

G.load_state_dict(torch.load('../DCGAN_14/checkpoints/G.ckpt'))


def denorm(x):
    out = (x * 0.5) + 0.5
    return out.clamp(0, 1)


# 存储生成的图像及其对应的tkinter兼容的图片对象
generated_images = []
tk_images = []


def generate_image():
    global generated_images, tk_images  # 声明全局变量

    # 如果已生成8张图片，清空列表和画布
    if len(generated_images) == 8:
        generated_images.clear()
        tk_images.clear()
        canvas.delete("all")

    # 生成新图片
    sample_vector = torch.randn(1, latent_size, 1, 1).to(device)
    with torch.no_grad():
        y = G(sample_vector)
        gen_img = denorm(y).cpu()
    img = gen_img[0].permute(1, 2, 0).numpy()
    img = Image.fromarray((img * 255).astype('uint8'))
    img_tk = ImageTk.PhotoImage(image=img)

    generated_images.append(img)
    tk_images.append(img_tk)  # 存储tkinter兼容的图片对象
    update_canvas()


def update_canvas():
    x_padding = 80  # 水平方向的间距
    y_padding = 80  # 垂直方向的间距
    img_width = 180  # 图片的宽度
    img_height = 180  # 图片的高度

    canvas.delete("all")  # 清除画布上的所有元素
    for i, img_tk in enumerate(tk_images):
        x = x_padding + (i % 4) * (img_width + x_padding)
        y = y_padding + (i // 4) * (img_height + y_padding)
        canvas.create_image(x, y, anchor='nw', image=img_tk)
        canvas.image = img_tk


def save_image():
    if generated_images:
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files", "*.png")])
        if file_path:
            # 假设保存最后生成的图像
            generated_images[-1].save(file_path)


def clear_images():
    global generated_images, tk_images
    generated_images = []
    tk_images = []
    canvas.delete("all")


def on_enter(e, color):
    e.widget['background'] = color


def on_leave(e, color):
    e.widget['background'] = color


def placeholder_command():
    pass  # 这里没有实际的功能


def main():
    global canvas
    root = tk.Tk()
    root.title('GAN for Synthetic Image Generator')
    root.configure(bg='#f0f0f0')  # 设置窗口背景颜色

    # 侧边栏
    sidebar = tk.Frame(root, bg='#f0f0f0', width=200)
    sidebar.pack(fill=tk.Y, side=tk.LEFT)
    # 底部标签
    footer_label = tk.Label(sidebar, text="@Designed by Kevin", bg='#f0f0f0', font=("Helvetica", 10))
    footer_label.pack(side=tk.BOTTOM, pady=10)

    # 画布
    canvas = Canvas(root, width=1200, height=600, bg='white')
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

    # 按钮
    generate_button = Button(sidebar, text="Generate Image", command=generate_image, font=("Helvetica", 12),
                             bg="#4CAF50", fg="white")
    generate_button.pack(padx=10, pady=10)
    generate_button.bind("<Enter>", lambda e: on_enter(e, '#45bf55'))
    generate_button.bind("<Leave>", lambda e: on_leave(e, '#4CAF50'))

    super_resolution_button = Button(sidebar, text="Super Resolution", command=placeholder_command,
                                     font=("Helvetica", 12),
                                     bg="#A9A9A9", fg="white")
    super_resolution_button.pack(padx=10, pady=10)
    super_resolution_button.bind("<Enter>", lambda e: on_enter(e, '#BEBEBE'))
    super_resolution_button.bind("<Leave>", lambda e: on_leave(e, '#A9A9A9'))

    save_button = Button(sidebar, text="Save Image", command=save_image, font=("Helvetica", 12), bg="#008CBA",
                         fg="white")
    save_button.pack(padx=10, pady=10)
    save_button.bind("<Enter>", lambda e: on_enter(e, '#009CDE'))
    save_button.bind("<Leave>", lambda e: on_leave(e, '#008CBA'))

    clear_button = Button(sidebar, text="Clear Images", command=clear_images, font=("Helvetica", 12), bg="#f44336",
                          fg="white")
    clear_button.pack(padx=10, pady=10)
    clear_button.bind("<Enter>", lambda e: on_enter(e, '#f55353'))
    clear_button.bind("<Leave>", lambda e: on_leave(e, '#f44336'))

    root.mainloop()


if __name__ == '__main__':
    main()
