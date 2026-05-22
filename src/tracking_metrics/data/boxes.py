from dataclasses import dataclass


@dataclass(frozen=True)
class Box2D:
    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return self.width() * self.height()

    def is_valid(self) -> bool:
        return self.x2 >= self.x1 and self.y2 >= self.y1 and self.area() >= 0
