# Taken from https://github.com/zkkli/PSAQ-ViT/blob/d2c726fe33d6dd474a5f2e2e88d36a354c9d7a7d/generate_data.py#L20

class AttentionMap:

    def __init__(self, module):
        self.feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        self.feature = None


def register_attention_hooks(model):
    """Register a hook on the attention output (attn @ v) of every block.

    The `matmul2` module is injected into each attention block by
    teachers.prepare.inject_attn_matmuls.
    """
    return [AttentionMap(blk.attn.matmul2) for blk in model.blocks]


def remove_hooks(hooks):
    for h in hooks:
        h.remove()
