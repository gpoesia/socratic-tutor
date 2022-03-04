def get_ax_name(ax_full):
    """
    Get name of axiom from a string 'ax_full' specifying both axiom and what it's applied to
    """
    for i, c in enumerate(ax_full):
        if c == " ":
            return ax_full[:i]
    return ax_full


def get_ax_param(ax_full):
    """
    Get parameters of axiom from a string 'ax_full' specifying both axiom and what it's applied to
    """
    for i, c in enumerate(ax_full):
        if c == " ":
            return ax_full[i+1:]
    return ""


def remove_brackets(str_):
    """
    Remove brackets in front of and after str_
    """
    for i, c in enumerate(str_):
        if c != '[':
            break
    for j in range(len(str_)-1,-1,-1):
        if str_[j] != ']':
            break
    return str_[i:j+1]


def make_tuple(abs_str):
    """
    Given abs_str (e.g. '[[comm-assoc]-[eval-mul1]]'), retrun tuple of the involved axioms ('comm', 'assoc', 'eval', 'mul1')
    """
    return tuple(map(remove_brackets, abs_str.split('-')))


def make_abs_str(abs_tuple):
    """
    Given abs_tuple (e.g. 'comm', 'assoc', 'eval', 'mul1'), return string of the involved axioms ('[comm-assoc-eval-mul1]')
    """
    return '-'.join(abs_tuple)


def make_param_str(param_tuple):
    """
    Given param_tuple, return string that joins parameters together with semicolon (;)
    """
    return '; '.join(param_tuple)

def is_prefix(pre, whole):
    """
    Return whether 'pre' (tuple) is prefix of 'whole' (tuple)
    """
    if len(pre) >= len(whole):
        return False
    for i, elt in enumerate(pre):
        if elt != whole[i]:
            return False
    return True


def prefix_get(pre, list_whole):
    """
    Return list of elements in 'list_whole' such that 'pre' is a prefix of them
    """
    return [elt for elt in list_whole if is_prefix(pre, elt)]
