package com.github.fairsearch.deltr.models;

import com.google.common.collect.Iterables;
import org.apache.lucene.search.ScoreDoc;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public class DeltrDocImpl extends ScoreDoc implements DeltrDoc {

    private TreeMap<String, Double> features = new TreeMap<String, Double>();
    private boolean isProtected;
    private String protectedFeatureName;

    public DeltrDocImpl(int doc, float score, boolean isProtected) {
        super(doc, score);
        this.isProtected = isProtected;
    }

    public DeltrDocImpl(int doc, float score, int shardIndex, boolean isProtected) {
        super(doc, score, shardIndex);
        this.isProtected = isProtected;
    }

    public void set(String name, Double value, boolean isProtectedFeature) {
        set(name, value);

        if(isProtectedFeature) {
            this.protectedFeatureName = name;
        }
    }

    @Override
    public void set(String name, Double value) {
        this.features.put(name, value);
    }

    @Override
    public int id() {
        return this.doc;
    }

    @Override
    public double judgement() {
        return this.score;
    }

    @Override
    public void rejudge(double judgment) {
        this.score = (float)judgment;
    }

    @Override
    public int size() {
        return this.features.size();
    }

    @Override
    public List<String> keys() {
        return new ArrayList<String>(this.features.keySet());
    }

    @Override
    public List<Double> features() {
        return new ArrayList<Double>(this.features.values());
    }

    @Override
    public Double feature(int index) {
        return Iterables.get(this.features.values(), index);
    }

    @Override
    public Double feature(String name) {
        return this.features.get(name);
    }

    @Override
    public boolean isProtected() {
        return this.isProtected;
    }

    @Override
    public String protectedFeatureName() {
        return this.protectedFeatureName;
    }

    @Override
    public int protectedFeatureIndex() {
        int index = 0;
        while(this.features.keySet().iterator().hasNext()){
            String name = this.features.keySet().iterator().next();
            if(name.equals(protectedFeatureName())) {
                break;
            }
            index++;
        }
        return index;
    }

    @Override
    public String toString() {
        return String.format("%f - %b", score, isProtected());
    }
}
