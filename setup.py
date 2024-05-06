from setuptools import setup, find_packages

setup(
    name='llmcategorizer',
    version='1.0.0',
    author='Chris Soria',
    author_email='chrissoria@berkeley.edu',
    description='A package for categorizing columns of data and outputting responses in a CSV format',
    long_description='LLMCategorizer is a Python package for automatically categorizing columns of data and outputting responses in a CSV format.',
    long_description_content_type='text/markdown',
    url='https://github.com/chrissoria/LLMCategorizer',
    packages=['llmcategorizer'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'openai>=1.0',  # Ensure versions are compatible with Python 3.6+
        'pandas>=1.0',
        'numpy>=1.15'  
    ],
)
