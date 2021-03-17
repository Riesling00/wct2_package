from setuptools import setup

setup(name='wct2',
      version='0.1',
      description='Photorealistic Style Transfer via Wavelet Transforms.',
      url = 'https://github.com/Riesling00/wct2',
      author='Riesling',
      author_email='leiyu6969@gmail.com',
      license = 'MIT License',
      packages=['wct2'],
      platforms = 'any',
      install_requires=[
        'torch==0.4.1',
        'torchvision==0.2.0',
        'tqdm==4.21.0',
        'numpy==1.14.5',
        'Pillow==6.2.0',
        'opencv-python'
      ],
      zip_safe=False)