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
    Given abs_str (e.g. '[[comm-assoc]-[eval-mul1]]'), return tuple of the involved axioms ('comm', 'assoc', 'eval', 'mul1')
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


class Trie:
    def __init__(self, key=None):
        self.key = key
        self.is_term = False
        self.children = {}

    def find(self, keys):
        node = self
        for key in keys:
            node = node.children.get(key)
            if node is None:
                return None
        return node if node.is_term else None

    def add(self, keys):
        # find deepest
        node = self
        path_exists = True
        for i, key in enumerate(keys):
            if node.children.get(key) is None:
                path_exists = False
                break
            node = node.children[key]

        if path_exists:
            node.is_term = True
        else: # create remaining path
            old_child = Trie(keys[-1])
            old_child.is_term = True
            for j in range(len(keys)-2, i-1, -1):
                new_child = Trie(keys[j])
                new_child.children[old_child.key] = old_child
                old_child = new_child
            node.children[old_child.key] = old_child


def make_abs_trie(abstractions):
    """
    Convert abstractions into trie
    """
    if not isinstance(abstractions[0], tuple):
        abstractions = list(map(make_tuple, abstractions))
    # abstractions is now list of tuples

    trie = Trie()
    for abs in abstractions:
        trie.add(abs)
    return trie


if __name__ == "__main__":
    # trie = Trie()
    # trie.add(("my", "name", "is", "dumb"))
    # trie.add(("your", "name", "is"))
    # trie.add(("my", "age", "is", 13))
    # trie.add(("my", "name", "bruh"))
    # trie.add(("my", "name", "is"))
    # trie.add(("my", "name", "is", "very", "dumb"))

    # my = trie.children['my']
    # myname = my.children['name']
    # mynameis = myname.children['is']
    # myage = my.children['age']
    # your = trie.children['your']
    # yourname = your.children['name']

    # trie = Trie()
    # trie.add((1, 2))
    # trie.add((1, 3))
 
    # one = trie.children[1]

    import json

    with open("mathematical-abstractions/abstractions/IterAbsPair-8k.json") as f:
        abs = json.load(f)['axioms']
    trie = make_abs_trie(abs)

