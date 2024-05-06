from setuptools import setup, find_packages

setup(
    name='CategorAI',
    version='1.0.1',
    author='Chris Soria',
    author_email='chrissoria@berkeley.edu',
    description='A package for categorizing columns of data and outputing responses in a CSV format',
    long_description='CategorAI is a Python package for automatically categorizing columns of data and outputing responses in a CSV format.',
    long_description_content_type='text/markdown',
    url='https://github.com/chrissoria/CategorAI',
    packages=find_packages(),  # Automatically find packages in the current directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'openai',   # OpenAI API client library
        'pandas',   # Data manipulation library
        'numpy',    # Numerical computing library  
    ],
)
