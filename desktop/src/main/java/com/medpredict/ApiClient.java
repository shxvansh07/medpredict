package com.medpredict;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Thin HTTP client for the MedPredict FastAPI backend (see ../../backend/app.py).
 * All ML/NLP logic lives server-side; this class only does request/response
 * plumbing so the UI code can work with plain JSON.
 */
public class ApiClient {

    private final String baseUrl;
    // Force HTTP/1.1: the JDK client's default HTTP/2 upgrade attempt over
    // plain http:// can drop the request body against a plain HTTP/1.1
    // server like uvicorn, surfacing as a 422 "field required" with a null
    // body server-side even though the client sent one.
    private final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    public ApiClient(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public JSONArray listSymptoms() throws IOException, InterruptedException {
        return new JSONArray(get("/symptoms"));
    }

    public JSONObject matchSymptoms(String freeText) throws IOException, InterruptedException {
        JSONObject body = new JSONObject().put("text", freeText);
        return new JSONObject(post("/match-symptoms", body.toString()));
    }

    public JSONObject predict(JSONArray symptomKeys, int topK) throws IOException, InterruptedException {
        JSONObject body = new JSONObject()
                .put("symptoms", symptomKeys)
                .put("top_k", topK);
        return new JSONObject(post("/predict", body.toString()));
    }

    public JSONObject findHospitals(double lat, double lon, String disease, int limit)
            throws IOException, InterruptedException {
        JSONObject body = new JSONObject()
                .put("lat", lat)
                .put("lon", lon)
                .put("disease", disease)
                .put("limit", limit);
        return new JSONObject(post("/hospitals", body.toString()));
    }

    public boolean healthCheck() {
        try {
            get("/health");
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private String get(String path) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(Duration.ofSeconds(15))
                .GET()
                .build();
        return send(req);
    }

    private String post(String path, String jsonBody) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(Duration.ofSeconds(20))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        return send(req);
    }

    private String send(HttpRequest req) throws IOException, InterruptedException {
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            throw new IOException("MedPredict API error " + resp.statusCode() + ": " + resp.body());
        }
        return resp.body();
    }
}
