def settings_dict(x,y, dict=None):
    ret = {} if dict is None else dict
    for x,y in zip(x,y):
        ret[x] = to_number(y)
    return ret

def to_number(s):
    try:
        ret = float(s)
        return ret if not float(ret).is_integer() else int(s)
    except ValueError:
        pass
    return s
