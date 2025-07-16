class HubLoadable:
    @classmethod
    def load_from_hub(cls, name: str, branch: str):
        try:
            import atria_hub  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )

        dataset = cls(dataset_name=name, config_name=branch, **(build_kwargs or {}))

        # Build the split for the dataset
        dataset.build_split(
            split=split,
            preprocess_transform=preprocess_transform,
            access_token=access_token,
            overwrite_existing_shards=overwrite_existing_shards,
            allowed_keys=allowed_keys,
            streaming=streaming,
            shard_storage_type=shard_storage_type,
            **(sharded_storage_kwargs if sharded_storage_kwargs else {}),
        )

        return cast(AtriaDataset[T_BaseDataInstance], dataset)

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str | None = None,
        is_public: bool = False,
    ) -> None:
        try:
            from atria_hub.api.datasets import DataInstanceType
            from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

            if name is None:
                name = self._dataset_name
            if branch is None:
                branch = self._config_name

            logger.info(
                f"Uploading dataset {self.__class__.__name__} to hub with name {name} and config {branch}."
            )

            def data_model_to_instance_type(
                data_model: type[T_BaseDataInstance],
            ) -> DataInstanceType:
                if data_model == DocumentInstance:
                    return DataInstanceType.DOCUMENT_INSTANCE
                elif data_model == ImageInstance:
                    return DataInstanceType.IMAGE_INSTANCE
                else:
                    raise ValueError(f"Unsupported data model: {data_model}")

            hub = AtriaHub()
            dataset = hub.datasets.get_or_create(
                name=name,
                description=self.metadata.description,
                data_instance_type=data_model_to_instance_type(self.data_model),
                is_public=is_public,
            )
            hub.datasets.upload_files(
                dataset=dataset,
                branch=branch,
                dataset_files=self.get_dataset_files_from_dir(),
            )
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )
