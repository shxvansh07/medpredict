package com.medpredict.ui;

import com.medpredict.ApiClient;
import com.medpredict.AppState;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;
import java.util.function.Consumer;

/**
 * Symptom intake + prediction screen.
 *
 * Two ways to enter symptoms, both feeding the same checklist:
 *   1. Free text ("it burns when I pee and I'm exhausted") -> matched via the
 *      backend's embedding-based symptom matcher, which auto-checks the
 *      matching boxes below.
 *   2. Direct checklist selection/search, for precise control or to add
 *      symptoms the free-text pass missed.
 */
public class SymptomCheckerPane extends BorderPane {

    private final ApiClient api;
    private final AppState state;
    private final Consumer<Void> onFindHospitals;

    private final Map<String, CheckBox> checkboxByKey = new LinkedHashMap<>();
    private final FlowPane checklistPane = new FlowPane(8, 6);
    private final TextArea freeTextArea = new TextArea();
    private final Label unmatchedLabel = new Label();
    private final VBox resultsBox = new VBox(10);
    private final Button findHospitalsBtn = new Button("Find Nearby Hospitals →");

    public SymptomCheckerPane(ApiClient api, AppState state, Consumer<Void> onFindHospitals) {
        this.api = api;
        this.state = state;
        this.onFindHospitals = onFindHospitals;
        buildUi();
        loadSymptomsAsync();
    }

    private void buildUi() {
        setPadding(new Insets(14));

        // --- Left: symptom intake -------------------------------------------------
        Label freeTextLabel = sectionLabel("Describe your symptoms in your own words");
        freeTextArea.setPromptText("e.g. it burns when I pee, I'm exhausted, and I have a high fever");
        freeTextArea.setPrefRowCount(3);
        freeTextArea.setWrapText(true);

        Button matchBtn = new Button("Match Symptoms From Text");
        matchBtn.setOnAction(e -> onMatchText());

        unmatchedLabel.setWrapText(true);
        unmatchedLabel.setStyle("-fx-text-fill: #a15c00;");

        TextField filterField = new TextField();
        filterField.setPromptText("Filter checklist (e.g. \"pain\")");
        filterField.textProperty().addListener((obs, old, val) -> applyFilter(val));

        checklistPane.setPadding(new Insets(8));
        ScrollPane checklistScroll = new ScrollPane(checklistPane);
        checklistScroll.setFitToWidth(true);
        checklistScroll.setPrefViewportHeight(320);

        Button predictBtn = new Button("Analyze Symptoms →");
        predictBtn.setDefaultButton(true);
        predictBtn.setOnAction(e -> onPredict());

        VBox left = new VBox(8,
                freeTextLabel, freeTextArea, matchBtn, unmatchedLabel,
                new Separator(),
                sectionLabel("Or select symptoms directly"),
                filterField, checklistScroll,
                predictBtn);
        left.setPrefWidth(430);
        left.setPadding(new Insets(0, 14, 0, 0));

        // --- Right: results ---------------------------------------------------------
        resultsBox.setPadding(new Insets(8));
        resultsBox.getChildren().add(new Label("Results will appear here after you analyze your symptoms."));
        ScrollPane resultsScroll = new ScrollPane(resultsBox);
        resultsScroll.setFitToWidth(true);

        findHospitalsBtn.setDisable(true);
        findHospitalsBtn.setOnAction(e -> onFindHospitals.accept(null));

        VBox right = new VBox(10, sectionLabel("Likely Conditions"), resultsScroll, findHospitalsBtn);
        VBox.setVgrow(resultsScroll, Priority.ALWAYS);

        SplitPane split = new SplitPane(left, right);
        split.setDividerPositions(0.48);
        setCenter(split);
    }

    private Label sectionLabel(String text) {
        Label l = new Label(text);
        l.setFont(Font.font("System", FontWeight.BOLD, 13));
        return l;
    }

    private void loadSymptomsAsync() {
        new Thread(() -> {
            try {
                JSONArray symptoms = api.listSymptoms();
                Platform.runLater(() -> {
                    for (int i = 0; i < symptoms.length(); i++) {
                        JSONObject s = symptoms.getJSONObject(i);
                        String key = s.getString("key");
                        String phrase = s.getString("phrase");
                        CheckBox cb = new CheckBox(phrase);
                        cb.setUserData(key);
                        checkboxByKey.put(key, cb);
                        checklistPane.getChildren().add(cb);
                    }
                });
            } catch (Exception ex) {
                Platform.runLater(() -> showError("Could not load symptom list: " + ex.getMessage()));
            }
        }, "load-symptoms").start();
    }

    private void applyFilter(String filter) {
        String f = filter == null ? "" : filter.toLowerCase();
        for (CheckBox cb : checkboxByKey.values()) {
            boolean visible = f.isEmpty() || cb.getText().toLowerCase().contains(f);
            cb.setVisible(visible);
            cb.setManaged(visible);
        }
    }

    private void onMatchText() {
        String text = freeTextArea.getText();
        if (text == null || text.isBlank()) return;

        new Thread(() -> {
            try {
                JSONObject result = api.matchSymptoms(text);
                Platform.runLater(() -> applyMatchResult(result));
            } catch (Exception ex) {
                Platform.runLater(() -> showError("Symptom matching failed: " + ex.getMessage()));
            }
        }, "match-symptoms").start();
    }

    private void applyMatchResult(JSONObject result) {
        JSONArray matched = result.getJSONArray("matched");
        for (int i = 0; i < matched.length(); i++) {
            JSONObject m = matched.getJSONObject(i);
            String symptom = m.getString("symptom");
            CheckBox cb = checkboxByKey.get(symptom);
            if (cb != null) cb.setSelected(true);
        }
        JSONArray unmatched = result.getJSONArray("unmatched_segments");
        if (unmatched.length() > 0) {
            List<String> segs = new ArrayList<>();
            for (int i = 0; i < unmatched.length(); i++) segs.add(unmatched.getString(i));
            unmatchedLabel.setText("Couldn't confidently match: " + String.join("; ", segs));
        } else {
            unmatchedLabel.setText("");
        }
    }

    private void onPredict() {
        JSONArray selected = new JSONArray();
        for (var entry : checkboxByKey.entrySet()) {
            if (entry.getValue().isSelected()) selected.put(entry.getKey());
        }
        if (selected.isEmpty()) {
            showError("Select or describe at least one symptom first.");
            return;
        }

        new Thread(() -> {
            try {
                JSONObject result = api.predict(selected, 3);
                Platform.runLater(() -> renderResults(result));
            } catch (Exception ex) {
                Platform.runLater(() -> showError("Prediction failed: " + ex.getMessage()));
            }
        }, "predict").start();
    }

    private void renderResults(JSONObject result) {
        resultsBox.getChildren().clear();
        JSONArray predictions = result.getJSONArray("predictions");

        for (int i = 0; i < predictions.length(); i++) {
            JSONObject p = predictions.getJSONObject(i);
            String disease = p.getString("disease");
            double confidence = p.getDouble("confidence");
            String specialty = p.getString("recommended_specialty");

            Label title = new Label((i + 1) + ". " + disease);
            title.setFont(Font.font("System", FontWeight.BOLD, 14));

            ProgressBar bar = new ProgressBar(confidence);
            bar.setPrefWidth(220);
            Label confLabel = new Label(String.format("%.0f%% confidence", confidence * 100));

            HBox confRow = new HBox(8, bar, confLabel);
            confRow.setAlignment(Pos.CENTER_LEFT);

            Label specialtyLabel = new Label("Suggested specialty: " + specialty.replace('_', ' '));
            specialtyLabel.setStyle("-fx-font-style: italic; -fx-text-fill: #555;");

            VBox explanationBox = new VBox(2);
            explanationBox.getChildren().add(new Label("Why:"));
            JSONArray explanation = p.getJSONArray("explanation");
            for (int j = 0; j < explanation.length(); j++) {
                JSONObject e = explanation.getJSONObject(j);
                explanationBox.getChildren().add(new Label("  • " + e.getString("phrase")));
            }

            VBox card = new VBox(4, title, confRow, specialtyLabel, explanationBox);
            card.setPadding(new Insets(10));
            card.setStyle("-fx-border-color: #ccc; -fx-border-radius: 4; -fx-background-color: #fafafa;");
            resultsBox.getChildren().add(card);
        }

        if (predictions.length() > 0) {
            JSONObject top = predictions.getJSONObject(0);
            state.lastPredictedDisease = top.getString("disease");
            state.lastRecommendedSpecialty = top.getString("recommended_specialty");
            findHospitalsBtn.setDisable(false);
        }
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR, message, ButtonType.OK);
        alert.showAndWait();
    }
}
