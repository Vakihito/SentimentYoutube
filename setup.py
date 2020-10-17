from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    version='0.0.1',
    name='SentimentYoutube',
    description='SentimentYoutube is a library that describes the feeling in through out the video, using keras',
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Aprroved :: MIT License',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    url='https://github.com/Vakihito/SentimentYoutube',
    author='Vakihito',
    athor_email='victor.kamada.tomita@gmail.com',
    keywords = ['tensorflow', 'keras', 'deep learning', 'machine learning', 'Youtube'],

    license='MIT',
    pakages=['SentimentYoutube'],
    install_requires=[
            'tensorflow==2.3.1',
            'ninja==1.10.0',
            'yacs==0.1.8',
            'cython==0.29.21',
            'matplotlib==3.3.2',    
            'demjson==2.2.4',    
            'fasttext==0.9.1',    
            'opencv-python==4.4.0.44',    
            'pytube3==9.6.0',    
            'textblob==0.15.3',    
            'scikit-learn>=0.21.3', # previously pinned to 0.21.3 due to TextPredictor.explain, but no longer needed as of 0.19.7
            'pandas >= 1.0.1',
            'fastprogress >= 0.1.21',
            'keras_bert>=0.86.0', # support for TF 2.3
            'requests',
            'joblib',
            'langdetect',
            'jieba',
            'cchardet',  # previously pinned to 2.1.5 (due to this issue: https://github.com/PyYoshi/cChardet/issues/61) but no longer needed
            'networkx>=2.3',
            'bokeh',
            'seqeval==0.0.19', # pin to 0.0.19 due to numpy version incompatibility with TensorFlow 2.3
            'packaging',
            'transformers>=3.1.0', # due to breaking change in v3.1.0
            'ipython',
            'syntok',
            'whoosh',
            'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI',
            'git+git://github.com/Vakihito/ktrain.git',
            'git+git://github.com/amaiya/eli5@tfkeras_0_10_1',
            'git+git://github.com/amaiya/stellargraph@no_tf_dep_082',
            'git+git://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI',
      ],

    
    
)

