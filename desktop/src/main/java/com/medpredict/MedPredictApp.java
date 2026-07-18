package com.medpredict;

import com.medpredict.ui.HospitalLocatorPane;
import com.medpredict.ui.SymptomCheckerPane;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.stage.Stage;

/**
 * MedPredict -- ML symptom checker and hospital locator.
 *
 * Entry point only: this class wires the two screens together over a shared
 * {@link AppState} and points {@link ApiClient} at the local backend
 * (see backend/app.py, run separately with `uvicorn app:app`).
 */
public class MedPredictApp extends Application {

    private static final String API_BASE_URL = "http://127.0.0.1:8000";

    @Override
    public void start(Stage stage) {
        ApiClient api = new ApiClient(API_BASE_URL);

        if (!api.healthCheck()) {
            Alert alert = new Alert(Alert.AlertType.WARNING,
                    "Could not reach the MedPredict backend at " + API_BASE_URL + ".\n\n" +
                    "Start it first:\n  cd backend && uvicorn app:app --port 8000\n\n" +
                    "The app will still open, but requests will fail until the backend is running.",
                    ButtonType.OK);
            alert.showAndWait();
        }

        AppState state = new AppState();
        TabPane tabs = new TabPane();
        tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

        HospitalLocatorPane hospitalPane = new HospitalLocatorPane(api, state);
        SymptomCheckerPane symptomPane = new SymptomCheckerPane(api, state, unused -> {
            tabs.getSelectionModel().select(1);
            hospitalPane.prefillFromPrediction();
        });

        Tab symptomTab = new Tab("Symptom Checker", symptomPane);
        Tab hospitalTab = new Tab("Hospital Locator", hospitalPane);
        tabs.getTabs().addAll(symptomTab, hospitalTab);

        Scene scene = new Scene(tabs, 1100, 720);
        stage.setTitle("MedPredict — Symptom Checker & Hospital Locator");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
