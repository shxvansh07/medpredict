package com.medpredict.ui;

import com.medpredict.ApiClient;
import com.medpredict.AppState;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Ranks nearby hospitals by haversine distance, boosting hospitals whose
 * specialty department matches the (optionally supplied) predicted condition
 * -- see backend's /hospitals endpoint for the actual ranking logic.
 *
 * Location is entered as latitude/longitude rather than pulled from a live
 * GPS/OS location API: a desktop JVM has no standard cross-platform GPS
 * source, so lat/lon input (defaulting to Bengaluru) keeps the geolocation
 * feature honest and testable without requiring platform-specific location
 * permissions.
 */
public class HospitalLocatorPane extends BorderPane {

    private final ApiClient api;
    private final AppState state;

    private final TextField latField = new TextField("12.9716");
    private final TextField lonField = new TextField("77.5946");
    private final TextField diseaseField = new TextField();
    private final Label specialtyLabel = new Label("Recommended specialty: —");
    private final TableView<HospitalRow> table = new TableView<>();
    private final ObservableList<HospitalRow> rows = FXCollections.observableArrayList();

    public HospitalLocatorPane(ApiClient api, AppState state) {
        this.api = api;
        this.state = state;
        buildUi();
    }

    private void buildUi() {
        setPadding(new Insets(14));

        GridPane form = new GridPane();
        form.setHgap(8);
        form.setVgap(8);
        form.addRow(0, new Label("Latitude:"), latField, new Label("Longitude:"), lonField);
        form.addRow(1, new Label("Condition (optional):"), diseaseField);

        Button searchBtn = new Button("Find Nearby Hospitals");
        searchBtn.setDefaultButton(true);
        searchBtn.setOnAction(e -> onSearch());

        HBox actions = new HBox(10, searchBtn, specialtyLabel);

        buildTable();

        VBox top = new VBox(10, new Label("Enter your location (defaults to central Bengaluru):"),
                form, actions);
        setTop(top);
        setCenter(table);
        BorderPane.setMargin(table, new Insets(12, 0, 0, 0));
    }

    private void buildTable() {
        TableColumn<HospitalRow, String> nameCol = new TableColumn<>("Hospital");
        nameCol.setCellValueFactory(new PropertyValueFactory<>("name"));
        nameCol.setPrefWidth(260);

        TableColumn<HospitalRow, String> distCol = new TableColumn<>("Distance (km)");
        distCol.setCellValueFactory(new PropertyValueFactory<>("distance"));
        distCol.setPrefWidth(110);

        TableColumn<HospitalRow, String> specialtiesCol = new TableColumn<>("Specialties");
        specialtiesCol.setCellValueFactory(new PropertyValueFactory<>("specialties"));
        specialtiesCol.setPrefWidth(320);

        TableColumn<HospitalRow, String> matchCol = new TableColumn<>("Recommended Match");
        matchCol.setCellValueFactory(new PropertyValueFactory<>("match"));
        matchCol.setPrefWidth(150);

        table.getColumns().setAll(nameCol, distCol, specialtiesCol, matchCol);
        table.setItems(rows);
        table.setPlaceholder(new Label("Search to see nearby hospitals."));
    }

    /** Called by MedPredictApp when the user clicks "Find Nearby Hospitals" from a prediction. */
    public void prefillFromPrediction() {
        if (state.lastPredictedDisease != null) {
            diseaseField.setText(state.lastPredictedDisease);
        }
        onSearch();
    }

    private void onSearch() {
        double lat, lon;
        try {
            lat = Double.parseDouble(latField.getText().trim());
            lon = Double.parseDouble(lonField.getText().trim());
        } catch (NumberFormatException ex) {
            showError("Latitude/longitude must be numbers.");
            return;
        }
        String disease = diseaseField.getText() == null || diseaseField.getText().isBlank()
                ? null : diseaseField.getText().trim();

        new Thread(() -> {
            try {
                JSONObject result = api.findHospitals(lat, lon, disease, 10);
                Platform.runLater(() -> renderResults(result));
            } catch (Exception ex) {
                Platform.runLater(() -> showError("Hospital search failed: " + ex.getMessage()));
            }
        }, "find-hospitals").start();
    }

    private void renderResults(JSONObject result) {
        rows.clear();
        String recommended = result.isNull("recommended_specialty") ? null : result.optString("recommended_specialty", null);
        specialtyLabel.setText("Recommended specialty: " + (recommended == null ? "— (no condition given)" : recommended.replace('_', ' ')));

        JSONArray hospitals = result.getJSONArray("hospitals");
        for (int i = 0; i < hospitals.length(); i++) {
            JSONObject h = hospitals.getJSONObject(i);
            StringBuilder specialties = new StringBuilder();
            JSONArray sp = h.getJSONArray("specialties");
            for (int j = 0; j < sp.length(); j++) {
                if (j > 0) specialties.append(", ");
                specialties.append(sp.getString(j).replace('_', ' '));
            }
            rows.add(new HospitalRow(
                    h.getString("name"),
                    String.format("%.2f", h.getDouble("distance_km")),
                    specialties.toString(),
                    h.getBoolean("matches_recommended_specialty") ? "✓ Yes" : ""
            ));
        }
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR, message, ButtonType.OK);
        alert.showAndWait();
    }

    /** Plain row POJO for the TableView (PropertyValueFactory needs public getters named getX()). */
    public static class HospitalRow {
        private final String name, distance, specialties, match;

        public HospitalRow(String name, String distance, String specialties, String match) {
            this.name = name;
            this.distance = distance;
            this.specialties = specialties;
            this.match = match;
        }

        public String getName() { return name; }
        public String getDistance() { return distance; }
        public String getSpecialties() { return specialties; }
        public String getMatch() { return match; }
    }
}
