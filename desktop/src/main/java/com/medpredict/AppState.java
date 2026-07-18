package com.medpredict;

/**
 * Small piece of state shared between the Symptom Checker and Hospital
 * Locator tabs: after a prediction, "Find Nearby Hospitals" hands the top
 * predicted disease + recommended specialty over to the locator tab.
 */
public class AppState {
    public volatile String lastPredictedDisease;
    public volatile String lastRecommendedSpecialty;
}
