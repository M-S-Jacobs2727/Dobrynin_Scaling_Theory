import torch

from Inception3 import Inception3
from old_Inception3 import Inception3 as old_Inception3


def main():
    newmodel = Inception3()
    newdict: dict[str, torch.Tensor] = newmodel.state_dict()
    oldmodel = old_Inception3()
    olddict: dict[str, torch.Tensor] = oldmodel.state_dict()

    newsizes = [p.shape for p in newdict.values()]
    oldsizes = [p.shape for p in olddict.values()]

    assert newsizes == oldsizes

    old2new = {o: n for o, n in zip(olddict, newdict)}
    for o, n in old2new.items():
        assert olddict[o].shape == newdict[n].shape
        newdict[n] = olddict[o]
    
    for o, n in zip(olddict.values(), newdict.values()):
        assert torch.all(o == n)
    
    newmodel.load_state_dict(newdict)

    a = torch.rand((1, 1, 224, 224))

    o: torch.Tensor = oldmodel(a)
    n: torch.Tensor = newmodel(a)

    print(o.shape, o)
    print(n.shape, n)


if __name__ == "__main__":
    main()