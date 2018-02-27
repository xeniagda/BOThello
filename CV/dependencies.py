import inspect

# name: value_in_tuple
_STATIC_VARS = {}
# name: (cached_value_in_tuple, func, [args])
_DYNAMIC_VARS = {}

def _remove(name):
    if name in _STATIC_VARS:
        del(_STATIC_VARS[name])
    if name in _DYNAMIC_VARS:
        del(_DYNAMIC_VARS[name])

def set_value(name, value):
    # print(name, "<-", value)


    updating = name in _STATIC_VARS
    _remove(name)
    _STATIC_VARS[name] = (value,)

    if updating:
        for i, dependent_name in enumerate(_DYNAMIC_VARS):
            _, _, args = _DYNAMIC_VARS[dependent_name]
            if name in args:
                # print(name, "removes", dependent_name)
                _invalidate(dependent_name)

def placeholder(name):
    _remove(name)
    _STATIC_VARS[name] = ()

def _invalidate(name):
    # print("REMOVE", name)
    _, func, args = _DYNAMIC_VARS[name]

    _remove(name)
    _DYNAMIC_VARS[name] = ((), func, args)

    for i, dependent_name in enumerate(_DYNAMIC_VARS):
        cached, func, args = _DYNAMIC_VARS[dependent_name]
        if name in args:
            # print(name, "removes", dependent_name)
            _invalidate(dependent_name)

def recalc(name):
    if name in _DYNAMIC_VARS:
        _invalidate(name)
        for arg in _DYNAMIC_VARS[name][2]:
            recalc(arg)

def get_value(name):
    # print("GET", name)
    if name in _STATIC_VARS:
        value = _STATIC_VARS[name]
        if value == ():
            raise RuntimeError("{} is only a placeholder".format(name))
        else:
            return value[0]
    elif name in _DYNAMIC_VARS:
        value, func, args = _DYNAMIC_VARS[name]
        if value != ():
            return value[0]
        else:
            arg_values = list(map(get_value, args))
            value = func(*arg_values)
            _DYNAMIC_VARS[name] = ((value,), func, args)
            return value
    else:
        raise KeyError("{} not found".format(name))

def is_placeholder(name):
    if name in _STATIC_VARS:
        return _STATIC_VARS[name] == ()
    elif name in _DYNAMIC_VARS:
        _, _, args = _DYNAMIC_VARS[name]
        return any(map(is_placeholder, args))
    else:
        raise KeyError("{} not found".format(name))

def exists(name):
    if name in _STATIC_VARS:
        return True
    elif name in _DYNAMIC_VARS:
        _, _, args = _DYNAMIC_VARS[name]
        return all(map(exists, args))
    else:
        return False

def dynamic(name):
    _remove(name)

    def wrapper(func):
        args = inspect.getargspec(func)[0]
        args_not_existing = list(filter(lambda x: not exists(x), args))
        if args_not_existing != []:
            raise KeyError("Arguments(s) to the function are not defined: {}".format(", ".join(args_not_existing)))
        # print(name, "<- f", args)
        _DYNAMIC_VARS[name] = ((), func, args)


    return wrapper

def draw_dependencies():
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()

    names = []

    for name in _DYNAMIC_VARS:
        g.add_node(len(names))
        names.append(name)

    for name in _STATIC_VARS:
        g.add_node(len(names))
        names.append(name)

    for name in _DYNAMIC_VARS:
        _, _, args = _DYNAMIC_VARS[name]
        for arg in args:
            g.add_edge(names.index(arg), names.index(name))


    labels = {}
    for i, name in enumerate(names):
        labels[i] = name

    print(labels)

    pos = nx.spring_layout(g)

    nx.draw(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=10)
    plt.show()


if __name__ == "__main__":
    # z = x + y - 2
    # w = y + 2
    # a = z - w + b
    placeholder("x")
    set_value("y", 3)
    set_value("b", 10)

    @dynamic("z")
    def update_z(x, y):
        # print("Updating z with x =", x, "y =", y)
        return x + y - 2

    @dynamic("w")
    def update_w(y):
        # print("Updating w with y =", y)
        return y + 2

    @dynamic("a")
    def update_a(z, w, b):
        # print("Updating a with z =", z, "w =", w, "b =", b)
        return z - w + b

    set_value("x", 5)

    print("Value of z =", get_value("z"))

    set_value("x", 2)
    print("Value of w =", get_value("w"))

    set_value("b", 2)
    print("Value of a =", get_value("a"))

    draw_dependencies()
