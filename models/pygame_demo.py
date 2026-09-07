"""
pygame_demo.py

Visual "object control" demo. Replays a subject's recording as a fake
live EEG stream (via stream_simulator.py) and moves a square on screen
according to the confirmed, decision-smoothed predictions:

    LEFT / RIGHT  -> move horizontally
    FORWARD       -> move up
    STOP          -> stop moving

This is a simulation, not a real-time BCI -- there is no EEG device
involved. It is meant to demonstrate the prediction-to-control pipeline
end to end: stream -> sliding-window classification -> smoothing ->
command -> visible action.

Requires: pip install pygame

Usage:
    python pygame_demo.py
"""

import os
import threading
import queue

import pygame

from stream_simulator import StreamSimulator, build_fake_continuous_stream

WIDTH, HEIGHT = 800, 500
SQUARE_SIZE = 40
SPEED = 4  # pixels per frame while a command is active

BG_COLOR = (18, 41, 77)      # matches the report/deck navy
SQUARE_COLOR = (46, 196, 182)  # matches the report/deck mint
TEXT_COLOR = (230, 236, 242)

command_queue = queue.Queue()


def on_command(label, name, command, t):
    """Called by StreamSimulator whenever a NEW confirmed command occurs."""
    print(f"[t={t:5.1f}s] {name} -> {command}")
    command_queue.put(command)


def run_simulator_in_background(data_folder, label_folder, model_path, subject, n_trials, speed_factor):
    X_continuous = build_fake_continuous_stream(data_folder, label_folder, subject, n_trials=n_trials)
    sim = StreamSimulator(
        model_path=model_path,
        window_seconds=4.0,
        step_seconds=0.5,
        smoothing_window=3,
        min_agreement=2,
        on_command=on_command,
    )
    # realtime=True + speed_factor lets the demo run at a watchable pace
    sim.run(X_continuous, realtime=True, speed_factor=speed_factor)


def main():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "BCI_IV_2a")
    LABEL_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "true_labels")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "single_subject_A01_csp12_svm.joblib")
    SUBJECT = "A01"
    N_TRIALS = 10
    SPEED_FACTOR = 4.0  # replay 4x faster than real time so the demo doesn't take forever to watch

    sim_thread = threading.Thread(
        target=run_simulator_in_background,
        args=(DATA_FOLDER, LABEL_FOLDER, MODEL_PATH, SUBJECT, N_TRIALS, SPEED_FACTOR),
        daemon=True,
    )
    sim_thread.start()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EEG Object Control -- Simulation (no hardware)")
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    x, y = WIDTH // 2, HEIGHT // 2
    current_command = "STOP"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Pick up the latest confirmed command, if any arrived
        while not command_queue.empty():
            current_command = command_queue.get()

        if current_command == "LEFT":
            x -= SPEED
        elif current_command == "RIGHT":
            x += SPEED
        elif current_command == "FORWARD":
            y -= SPEED
        # STOP -> no movement

        x = max(0, min(WIDTH - SQUARE_SIZE, x))
        y = max(0, min(HEIGHT - SQUARE_SIZE, y))

        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, SQUARE_COLOR, (x, y, SQUARE_SIZE, SQUARE_SIZE), border_radius=8)

        label = font.render(f"Current command: {current_command}", True, TEXT_COLOR)
        screen.blit(label, (20, 20))

        note = font.render("Simulated stream (replayed recording) -- no EEG device", True, (159, 180, 204))
        screen.blit(note, (20, HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
