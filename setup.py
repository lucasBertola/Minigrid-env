from setuptools import setup, find_packages


setup(
    name='gymnasium_minigrid',
    version='0.0.5',
    description='A mini grid environment for OpenAI Gym and Gymnasium',
    author='Lucas Bertola',
    url='https://github.com/lucasBertola/Minigrid',  
    # author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pygame==2.1.3',
        'gymnasium==0.28.1',
    ]
)

#python setup.py sdist bdist_wheel
#twine upload dist/*