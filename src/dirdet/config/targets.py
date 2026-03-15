from dataclasses import dataclass

@dataclass(frozen=True) # prevent changin attributes of instances later
class Atom:
    name: str
    A: int  # Mass Number
    Z: int  # Atomic Number
    
    @property
    def N(self) -> int:
        # neutron number
        return self.A - self.Z

    
FLUORINE = Atom(name="Fluorine",A=19,Z=9)
XENON = Atom(name="Xenon",A=131,Z=54)