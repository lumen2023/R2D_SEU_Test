from . import parallel, wrappers


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def make_envs(config):
    def env_constructor(split):
        return lambda idx: lambda: make_env(config, idx, split)

    train_envs = parallel.ParallelEnv(env_constructor("train"), config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor("eval"), config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id, split="train"):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    elif suite == "metadrive":
        import envs.metadrive as metadrive

        if split == "train":
            start_seed = _get(config, "train_start_seed", _get(config, "start_seed", None))
            num_scenarios = _get(config, "train_num_scenarios", _get(config, "num_scenarios", None))
        else:
            start_seed = _get(config, "eval_start_seed", _get(config, "start_seed", None))
            num_scenarios = _get(config, "eval_num_scenarios", _get(config, "num_scenarios", None))
        env = metadrive.MetaDrive(
            task,
            action_repeat=config.action_repeat,
            seed=config.seed + id,
            split=split,
            metadrive_source=_get(config, "metadrive_source", None),
            num_scenarios=num_scenarios,
            start_seed=start_seed,
            extra_config=_get(config, "extra_config", None),
        )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
