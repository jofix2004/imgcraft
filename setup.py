# /setup.py
from setuptools import setup, find_packages

setup(
    name='imgcraft',
    version='0.1.0',
    author='[Tên của bạn]',
    description='Một tiện ích xử lý và chỉnh sửa ảnh dựa trên ComfyUI và mô hình FLUX.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'torch==2.6.0',
        'torchvision==0.21.0',
        'numpy',
        'Pillow',
        'imageio',
        'imageio-ffmpeg',
        'requests',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
