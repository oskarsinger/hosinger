from distutils.core import setup

setup(
    name='OskarResearchCode',
    version='0.01',
    packages=[
        'optimization',
        'optimization.optimizers',
        'optimization.optimizers.ftprl',
        'linal',
        'linal.random',
        'linal.utils',
        'lazyprojector',
        'drrobert',
        'drrobert.data_structures',
        'data',
        'data.errors',
        'data.loaders',
        'data.loaders.e4',
        'data.loaders.random',
        'data.servers',
        'data.servers.action_reward',
        'data.servers.batch',
        'data.servers.minibatch',
        'data.servers.gram'])
