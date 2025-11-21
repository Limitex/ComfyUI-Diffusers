class CreateIntListNode:
    """Utility node to create a list of integers for Stream Diffusion t_index_list."""

    MAX_ELEMENTS = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, int]] | tuple[str, ...]]]:
        inputs: dict[str, dict[str, tuple[str, dict[str, int]] | tuple[str, ...]]] = {
            "required": {
                "elements_count": (
                    "INT",
                    {"default": 2, "min": 1, "max": cls.MAX_ELEMENTS, "step": 1},
                ),
            },
            "optional": {},
        }
        for i in range(1, cls.MAX_ELEMENTS):
            inputs["optional"][f"element_{i}"] = ("INT", {"default": 0})
        return inputs

    RETURN_TYPES = ("LIST",)
    FUNCTION = "execute"
    CATEGORY = "Diffusers/StreamDiffusion"

    def execute(self, elements_count: int, **kwargs: int) -> tuple[list[int]]:
        """Create a list of integers from the input parameters.

        Args:
            elements_count: Number of elements to include in the list
            **kwargs: Optional element values

        Returns:
            Tuple containing the list of integers
        """
        result = [value for key, value in kwargs.items()][:elements_count]
        return (result,)
