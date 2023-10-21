def orthogonal_projection(A, B):
    out = A - B @ A.transpose(-2, -1) @ B
    return out


def retraction(A, ref=None):
    if ref is None:
        data = A
    else:
        data = A + ref
    Q, R = data.qr()
    sign = (R.diagonal(dim1=-2, dim2=-1).sign() + 0.5).sign().diag_embed()
    out = Q @ sign
    return out
