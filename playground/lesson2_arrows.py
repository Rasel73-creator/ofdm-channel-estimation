import numpy as np
import matplotlib.pyplot as plt

# 1. A channel value is an arrow: 0.6 to the right, 0.4 up
h = 0.6 + 0.4j
print("length of h (louder/quieter):", round(abs(h), 3))
print("angle of h in degrees (time shift):", round(np.degrees(np.angle(h)), 1))

# 2. Send the pilot (an arrow of length 1 pointing right) through the room
pilot = 1 + 0j
received = h * pilot

# 3. Add a small random arrow on top: the hiss
rng = np.random.default_rng(0)
noise = 0.1 * (rng.normal() + 1j * rng.normal())
z = received + noise
print("measured z:", np.round(z, 3))

print("h times 1j: ", round(abs(h * 1j), 3), round(np.degrees(np.angle(h * 1j)), 1))
print("h times 0.5:", round(abs(h * 0.5), 3), round(np.degrees(np.angle(h * 0.5)), 1))

# 4. Draw the arrows
def draw(arrow, colour, label, nudge=0.0):
    plt.arrow(0, 0, arrow.real, arrow.imag, color=colour,
              width=0.01, length_includes_head=True)
    plt.text(arrow.real, arrow.imag + nudge, "  " + label, color=colour)

draw(pilot, "grey", "pilot sent")
draw(h, "blue", "true channel h", nudge=0.04)
draw(z, "red", "measured z", nudge=-0.06)
plt.axis("equal")
plt.grid(True)
plt.xlabel("right (real part)")
plt.ylabel("up (imaginary part)")
plt.show()