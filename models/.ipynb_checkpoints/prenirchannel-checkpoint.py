import kornia as K

def edge_nir(x):
    print(x.shape)
    nir = x[:,3,:,:]
    magnitude_nir_target, edge_nir_target = K.filters.canny(nir)
    return y