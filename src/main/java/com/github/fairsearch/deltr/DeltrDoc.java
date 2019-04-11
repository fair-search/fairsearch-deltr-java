package com.github.fairsearch.deltr;

import org.apache.lucene.search.ScoreDoc;

import java.util.List;

public class DeltrDoc extends ScoreDoc {

    // Specifies if the document is protected or not
    public boolean isProtected;

    // list of features
    public List<Float> features;

    public DeltrDoc(int doc, float score) {
        super(doc, score);
    }

    public DeltrDoc(int doc, float score, int shardIndex) {
        super(doc, score, shardIndex);
    }

    public DeltrDoc(int doc, float score, boolean isProtected) {
        super(doc, score);
        this.isProtected = isProtected;
    }

    public DeltrDoc(int doc, float score, boolean isProtected, List<Float> features) {
        super(doc, score);
        this.isProtected = isProtected;
        this.features = features;
    }

    public DeltrDoc(int doc, float score, int shardIndex, boolean isProtected, List<Float> features) {
        super(doc, score, shardIndex);
        this.isProtected = isProtected;
        this.features = features;
    }

    @Override
    public String toString() {
        return "doc=" + doc + " score=" + score + " isProtected=" + isProtected;
    }
}
