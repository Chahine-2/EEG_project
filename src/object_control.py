import pygame
import time
import joblib
import numpy as np
import os


pygame.init()
font = pygame.font.Font(None, 36)


# Load trained EEG models
csp = joblib.load("models/csp.pkl")
svm = joblib.load("models/svm.pkl")


# Class → command mapping
commands = {
    7: "LEFT",
    8: "RIGHT",
    9: "FORWARD",
    10: "STOP"
}


# ==========================
# EEG Prediction
# ==========================

trials = np.load("data/test_trials.npy")

print("Trials shape:", trials.shape)


trial_index = 0


def get_command(index):

    trial = trials[index]

    # Add batch dimension
    trial = trial.reshape(1, 22, 1001)

    # CSP feature extraction
    features = csp.transform(trial)

    # SVM prediction
    prediction = svm.predict(features)

    command = commands[prediction[0]]

    print(
        "Trial:",
        index,
        "Class:",
        prediction[0],
        "Command:",
        command
    )

    return command



command = get_command(trial_index)



# ==========================
# Pygame
# ==========================

screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("EEG Object Control")

clock = pygame.time.Clock()

x = 300
y = 200

size = 40
speed = 5

from sklearn.metrics import accuracy_score


labels = np.load("data/test_labels.npy")

test_features = csp.transform(trials)

test_predictions = svm.predict(test_features)

accuracy = accuracy_score(
    labels,
    test_predictions
) * 100

# Timer
last_prediction_time = pygame.time.get_ticks()
prediction_delay = 2000   # 2 seconds

running = True

while running:

    # ---------------- Events ----------------
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    # --------- New EEG prediction ----------
    current_time = pygame.time.get_ticks()

    if current_time - last_prediction_time >= prediction_delay:

        trial_index += 1

        if trial_index >= len(trials):
            trial_index = 0

        command = get_command(trial_index)

        last_prediction_time = current_time

    # ------------- Move object -------------
    if command == "LEFT":
        x -= speed

    elif command == "RIGHT":
        x += speed

    elif command == "FORWARD":
        y -= speed

    elif command == "STOP":
        pass

    # -------- Screen wrapping --------
    if x < -size:
        x = 600

    if x > 600:
        x = -size

    if y < -size:
        y = 400

    if y > 400:
        y = -size

    # ---------- Command color ----------
    if command == "LEFT":
        color = (0, 150, 255)

    elif command == "RIGHT":
        color = (255, 255, 0)

    elif command == "FORWARD":
        color = (0, 255, 0)

    else:  # STOP
        color = (255, 0, 0)

    # --------------- Draw ---------------
    screen.fill((30, 30, 30))

    # Title
    title = font.render("EEG OBJECT CONTROL", True, (255, 255, 255))
    screen.blit(title, (150, 10))

    # Information
    command_text = font.render(f"Command : {command}", True, color)
    screen.blit(command_text, (10, 60))

    trial_text = font.render(
        f"Trial : {trial_index + 1}/{len(trials)}",
        True,
        (255, 255, 255)
    )
    screen.blit(trial_text, (10, 100))

    accuracy_text = font.render(
        f"Accuracy : {accuracy:.2f}%",
        True,
        (255, 255, 255)
    )
    screen.blit(accuracy_text, (10, 140))

    # Controlled object
    pygame.draw.rect(
        screen,
        (255, 255, 255),
        (x, y, size, size)
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()