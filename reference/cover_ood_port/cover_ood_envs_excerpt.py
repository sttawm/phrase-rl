class PutSpoonOnTableClothInSceneGoogle(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        source_obj_name="bridge_spoon_generated_modified",
        target_obj_name="table_cloth_generated_shorter",
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.05
        half_edge_length_y = 0.05
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super().evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return "put the spoon on the towel"
    
    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_coke_can_for_bridge.png"
        )
        return ret

@register_env("PutRedbullOnPlateInScene", max_episode_steps=60)
class PutRedbullOnPlateInScene(PutOnBridgeInSceneEnv, CustomBridgeObjectsInSceneEnvV1):
    def __init__(
        self,
        source_obj_name="redbull_can",
        target_obj_name="bridge_plate_objaverse_larger",
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])  # All corners and center
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )
        
        self.distractors = [
            ("coke_can",),
        ]
        self._distractor_ids = list(range(1))
        xy_configs = [
            [grid_pos[1], grid_pos[0], grid_pos[2]],  # source at bottom-left, target at top-right, distractor at top-left
            [grid_pos[1], grid_pos[0], grid_pos[3]],  # source at bottom-left, target at top-right, distractor at bottom-right
            [grid_pos[2], grid_pos[3], grid_pos[0]],  # source at top-right, target at bottom-left, distractor at top-left
            [grid_pos[2], grid_pos[3], grid_pos[1]],  # source at top-right, target at bottom-left, distractor at bottom-right
        ]

        quat_configs = [
            np.array([np.array(euler2quat(0, -np.pi / 2, np.pi / 2, 'sxyz')), [0, 0, 0, 1]]),
        ]

        self.distractor_init_quat_dict = {  # distractor has forced quaternions
            "coke_can": euler2quat(0, np.pi / 2, np.pi / 2, 'sxyz')
        }

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        
        # source and target object configs
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]
        
        # distractor configs
        _num_episodes = (
            len(self.distractors)
            * len(self._distractor_ids)
            * len(self._xy_configs)
        )
        episode_id = obj_init_options.get(
            "episode_id", self._episode_rng.randint(_num_episodes)
        )
        episode_id = episode_id % _num_episodes
        distractor_list = self.distractors[
            episode_id // (len(self._distractor_ids) * len(self._xy_configs))
        ]
        quat_config_distractor = [
            self.distractor_init_quat_dict[model_id] for model_id in distractor_list
        ]

        options["model_ids"] = [self._source_obj_name, self._target_obj_name] + list(distractor_list)
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = list(xy_config)
        obj_init_options["init_rot_quats"] = list(quat_config) + list(quat_config_distractor)
        options["obj_init_options"] = obj_init_options

        obs, info = PutOnInSceneEnv.reset(self, seed=self._episode_seed, options=options)  # Call Parent's Parent, ie. PutOnInSceneEnv
        info.update({"episode_id": episode_id})
        return obs, info

    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super().evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return "put redbull can on plate"

@register_env("PutTennisBallInBasketScene", max_episode_steps=120)
class PutTennisBallInBasketScene(PutOnBridgeInSceneEnv, CustomBridgeObjectsInSceneEnvV1):
    def __init__(
        self,
        **kwargs,
    ):
        source_obj_name = "tennis_ball"
        target_obj_name = "dummy_sink_target_plane"  # invisible
        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]

        # Increase spacing between objects to prevent overlap
        half_span_x = 0.02  # Increased from 0.01
        half_span_y = 0.03  # Increased from 0.015
        num_x = 2
        num_y = 3  # Reduced from 4 to create more space

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1]]))

        self.distractors = [
            ("orange", "pingpong_ball"),  # Both distractors appear together
        ]
        self._distractor_ids = list(range(2))
        xy_configs = [
            (pos[0], target_xy, *pos[1:]) 
            for pos in permutations(grid_pos, len(self._distractor_ids) + 1)
        ]
        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, -1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
        ]
        self.distractor_init_quat_dict = { # distractor has forced quaternions
            "orange": euler2quat(0, 0, np.pi / 2, 'sxyz'),
            "pingpong_ball": [1.0, 0.0, 0.0, 0.0],
        }
        self.special_density_dict = {
            "pingpong_ball": 200,  # toy apple as in real eval
            "orange": 200
            # by default, opened cans have density 50; blue plastic bottle has density 50; sponge has density 150
        }
        
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=['sink', 'dummy_sink_target_plane'],
            **kwargs,
        )
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        
        # source and target object configs
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]
        
        # distractor configs
        _num_episodes = (
            len(self.distractors)
            * len(self._distractor_ids)
            * len(self._xy_configs)
        )
        episode_id = obj_init_options.get(
            "episode_id", self._episode_rng.randint(_num_episodes)
        )
        episode_id = episode_id % _num_episodes
        distractor_list = self.distractors[0]  # Always use the tuple with both distractors
        quat_config_distractor = [
            self.distractor_init_quat_dict[model_id] for model_id in distractor_list
        ]

        options["model_ids"] = [self._source_obj_name, self._target_obj_name] + list(distractor_list)
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = list(xy_config)
        obj_init_options["init_rot_quats"] = list(quat_config) + list(quat_config_distractor)
        options["obj_init_options"] = obj_init_options

        obs, info = PutOnInSceneEnv.reset(self, seed=self._episode_seed, options=options) # Call Parent's Parent, ie. PutOnInSceneEnv
        info.update({"episode_id": episode_id})
        return obs, info

    def get_language_instruction(self, **kwargs):
        return "put tennis ball into yellow basket"

    def _load_model(self):
        super()._load_model()
        self.sink_id = 'sink'
        self.sink = self._build_actor_helper(
            self.sink_id,
            self._scene,
            density=self.model_db[self.sink_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.sink.name = self.sink_id

    def _initialize_actors(self):
        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self.sink.set_pose(sapien.Pose(
            [-0.16, 0.13, 0.88],
            [1, 0, 0, 0]
        ))
        self.sink.lock_motion()

        super()._initialize_actors()

    def evaluate(self, *args, **kwargs):
        return super().evaluate(success_require_src_completely_on_target=False, 
                                z_flag_required_offset=0.06,
                                *args, **kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_sink.png"
        )
        return ret

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.127, 0.06],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False # in env reset options, no need to reconfigure the environment

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )

@register_env("PutZucchiniOnTableClothInScene", max_episode_steps=60)
class PutZucchiniOnTableClothInScene(PutOnBridgeInSceneEnv, CustomBridgeObjectsInSceneEnvV1):
    def __init__(self, **kwargs):
        source_obj_name = "zucchini"
        target_obj_name = "bridge_plate_objaverse_blue" # "cool_plate"

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        self.distractors = [
            ("bridge_carrot_generated_modified",),
        ]
        self._distractor_ids = list(range(1))
        xy_configs = list(permutations(grid_pos, len(self._distractor_ids) + 2))

        quat_configs = [
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
        ]
        self.distractor_init_quat_dict = { # distractor has forced quaternions
            "bridge_carrot_generated_modified": np.array([1, 0, 0, 0])
        }

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )
        
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        
        # source and target object configs
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]
        
        # distractor configs
        _num_episodes = (
            len(self.distractors)
            * len(self._distractor_ids)
            * len(self._xy_configs)
        )
        episode_id = obj_init_options.get(
            "episode_id", self._episode_rng.randint(_num_episodes)
        )
        episode_id = episode_id % _num_episodes
        distractor_list = self.distractors[
            episode_id // (len(self._distractor_ids) * len(self._xy_configs))
        ]
        quat_config_distractor = [
            self.distractor_init_quat_dict[model_id] for model_id in distractor_list
        ]

        options["model_ids"] = [self._source_obj_name, self._target_obj_name] + list(distractor_list)
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = list(xy_config)
        obj_init_options["init_rot_quats"] = list(quat_config) + list(quat_config_distractor)
        options["obj_init_options"] = obj_init_options

        obs, info = PutOnInSceneEnv.reset(self, seed=self._episode_seed, options=options) # Call Parent's Parent, ie. PutOnInSceneEnv
        info.update({"episode_id": episode_id})
        return obs, info
    
    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super().evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return "put the zucchini on the towel"

@register_env("PutTapeMeasureInBasketScene-v0", max_episode_steps=120)
class PutTapeMeasureInBasketScene(PutOnBridgeInSceneEnv, CustomBridgeObjectsInSceneEnvV1):
    def __init__(self, **kwargs):
        source_obj_name = "tape_measure"
        target_obj_name = "dummy_sink_target_plane"  # invisible
        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.04
        half_span_y = 0.04
        num_x = 2
        num_y = 2

