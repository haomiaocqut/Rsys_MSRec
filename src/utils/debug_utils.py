
def get_gin_para(model):
    return model.gnn.ginlayers[0].apply_func.mlp.linears[0].weight[:3, :3]
