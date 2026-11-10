
import pygame
import sys


class StartupPredictionUI:

    def __init__(self, predictor, results, baseline, cv_results, records):
        pygame.init()

        self.screen = pygame.display.set_mode((1050, 850))
        pygame.display.set_caption("Startup Acquisition Prediction")

        self.predictor = predictor
        self.results = results
        self.baseline = baseline
        self.cv = cv_results
        self.records = records

        self.font = pygame.font.SysFont("Arial", 19)
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)

        self.funding = ""
        self.prediction = None
        self.probabilities = None
        self.show_probabilities = False

        self.clock = pygame.time.Clock()

    def draw_text(self, text, x, y, font=None, color=(230, 230, 230)):
        if font is None:
            font = self.font

        surface = font.render(str(text), True, color)
        self.screen.blit(surface, (x, y))

    def loading_screen(self):
        messages = [
            "Preparing startup prediction dashboard...",
            "Loading model evaluation results...",
            "Preparing prediction interface..."
        ]

        for message in messages:
            start = pygame.time.get_ticks()

            while pygame.time.get_ticks() - start < 700:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                self.screen.fill((20, 25, 40))

                self.draw_text(
                    "STARTUP ACQUISITION PREDICTION",
                    230, 300, self.title_font
                )

                self.draw_text(message, 300, 380)

                pygame.display.flip()
                self.clock.tick(60)

    def predict(self):
        try:
            amount = float(self.funding)

            if amount < 0:
                raise ValueError

            self.prediction = self.predictor.predict(amount)[0]
            self.probabilities = self.predictor.predict_proba(amount)[0]

        except ValueError:
            self.prediction = "Invalid funding amount"
            self.probabilities = None

    def draw(self):
        self.screen.fill((20, 25, 40))

        self.draw_text(
            "STARTUP ACQUISITION PREDICTION",
            30, 20, self.title_font
        )

        self.draw_text(
            f"Records: {self.records}",
            30, 65
        )

        self.draw_text("MODEL EVALUATION", 30, 110, self.title_font)

        metrics = [
            f"Accuracy: {self.results['accuracy']:.4f}",
            f"Balanced accuracy: {self.results['balanced_accuracy']:.4f}",
            f"Precision: {self.results['precision']:.4f}",
            f"Recall: {self.results['recall']:.4f}",
            f"F1-score: {self.results['f1']:.4f}",
            f"Baseline accuracy: {self.baseline:.4f}",
        ]

        for i, metric in enumerate(metrics):
            self.draw_text(metric, 30, 155 + i * 27)

        self.draw_text("CONFUSION MATRIX", 550, 110, self.title_font)

        matrix = self.results["confusion_matrix"]

        self.draw_text("                 Predicted", 550, 155)
        self.draw_text("                 0          1", 550, 185)
        self.draw_text(f"Actual 0       {matrix[0][0]}       {matrix[0][1]}", 550, 215)
        self.draw_text(f"Actual 1       {matrix[1][0]}       {matrix[1][1]}", 550, 245)

        self.draw_text("CLASSIFICATION REPORT", 30, 340, self.title_font)

        report = self.results["classification_report"].split("\n")

        for i, line in enumerate(report):
            self.draw_text(line, 30, 385 + i * 23)

        self.draw_text("CROSS-VALIDATION", 550, 340, self.title_font)

        self.draw_text(
            f"Fold scores: {self.cv['scores'].round(4)}",
            550, 385
        )

        self.draw_text(
            f"Mean accuracy: {self.cv['mean']:.4f}",
            550, 420
        )

        self.draw_text(
            f"Standard deviation: {self.cv['std']:.4f}",
            550, 455
        )

        self.draw_text("MAKE A PREDICTION", 550, 530, self.title_font)

        pygame.draw.rect(self.screen, (60, 70, 90), (550, 580, 250, 40))
        self.draw_text(self.funding or "Enter funding amount", 560, 590)

        pygame.draw.rect(self.screen, (40, 130, 90), (820, 580, 110, 40))
        self.draw_text("Predict", 845, 590)

        pygame.draw.rect(self.screen, (80, 90, 110), (550, 635, 110, 35))
        self.draw_text("Reset", 580, 642)

        checkbox = (690, 642, 25, 25)
        pygame.draw.rect(self.screen, (220, 220, 220), checkbox, 2)

        if self.show_probabilities:
            pygame.draw.rect(self.screen, (70, 190, 130), (695, 647, 15, 15))

        self.draw_text("Show probabilities", 725, 642)

        if self.prediction is not None:
            if self.prediction == 1:
                message = "Predicted: Acquired"
            elif self.prediction == 0:
                message = "Predicted: Not acquired"
            else:
                message = self.prediction

            self.draw_text(message, 550, 700, color=(100, 220, 150))

            if self.show_probabilities and self.probabilities is not None:
                self.draw_text(
                    f"Not acquired: {self.probabilities[0]:.2%}",
                    550, 735
                )

                self.draw_text(
                    f"Acquired: {self.probabilities[1]:.2%}",
                    550, 765
                )

        pygame.display.flip()

    def run(self):
        self.loading_screen()

        running = True

        while running:
            self.draw()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    x, y = event.pos

                    if 820 <= x <= 930 and 580 <= y <= 620:
                        self.predict()

                    elif 550 <= x <= 660 and 635 <= y <= 670:
                        self.funding = ""
                        self.prediction = None
                        self.probabilities = None

                    elif 690 <= x <= 715 and 635 <= y <= 670:
                        self.show_probabilities = not self.show_probabilities

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_BACKSPACE:
                        self.funding = self.funding[:-1]

                    elif event.key == pygame.K_RETURN:
                        self.predict()

                    elif event.unicode.isdigit() or event.unicode == ".":
                        self.funding += event.unicode

            self.clock.tick(60)

        pygame.quit()
        sys.exit()
