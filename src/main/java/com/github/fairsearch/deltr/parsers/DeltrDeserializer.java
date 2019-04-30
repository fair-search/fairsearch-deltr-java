package com.github.fairsearch.deltr.parsers;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.github.fairsearch.deltr.Deltr;

import java.io.IOException;

public class DeltrDeserializer extends JsonDeserializer<Deltr> {

    @Override
    public Deltr deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException, JsonProcessingException {
        ObjectCodec oc = jp.getCodec();
        JsonNode node = oc.readTree(jp);

        final double gamma = node.get("gamma").asDouble();
        final int numberOfIterations = node.get("number_of_iterations").asInt();
        final double learningRate = node.get("learning_rate").asDouble();
        final double lambda = node.get("lambda").asDouble();
        final double initVar = node.get("init_var").asDouble();
        final boolean shouldStandardize = node.get("standardize").asBoolean();

        final double mu = node.get("mu").asDouble();
        final double sigma = node.get("sigma").asDouble();

        double[] omega = new double[node.get("omega").size()];
        for(int i=0; i< node.get("omega").size(); i++) {
            omega[i] = node.get("omega").get(i).asDouble();
        }

        return new Deltr(gamma, numberOfIterations, learningRate, lambda, initVar, shouldStandardize, mu, sigma, omega);
    }
}
