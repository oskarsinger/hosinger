from distutils.core import setup

setup(
    name='OskarResearchCode',
    version='0.01',
    packages=[
        'em',
        'learners',
        'wavelets',
        'wavelets.dtcwt',
        'optimization',
        'optimization.optimizers',
        'optimization.optimizers.ftprl',
        'optimization.optimizers.quasinewton',
        'optimization.stepsize',
        'optimization.qnservers',
        'linal',
        'linal.tensor',
        'linal.random',
        'linal.utils',
        'lazyprojector',
        'drrobert',
        'drrobert.data_structures',
        'drrobert.fp',
        'data',
        'data.errors',
        'data.loaders',
        'data.loaders.e4',
        'data.loaders.at',
        'data.loaders.synthetic',
        'data.loaders.readers',
        'data.servers',
        'data.servers.action_reward',
        'data.servers.batch',
        'data.servers.minibatch',
        'data.servers.masks',
        'data.servers.gram',
        'runners',
        'runners.bandit'])
