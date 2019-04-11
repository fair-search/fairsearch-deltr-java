package com.github.fairsearch.deltr;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class DeltrTopDocsImpl extends TopDocs implements DeltrTopDocs {

    private int questionId;

    public DeltrTopDocsImpl(int questionId, long totalHits, ScoreDoc[] scoreDocs, float maxScore) {
        super(totalHits, scoreDocs, maxScore);

        this.questionId = questionId;
    }

    @Override
    public int id() {
        return this.questionId;
    }

    @Override
    public int size() {
        return this.scoreDocs.length;
    }

    @Override
    public List<DeltrDoc> docs() {
        return Arrays.stream(this.scoreDocs).map(x -> (DeltrDoc)x).collect(Collectors.toList());
    }

    @Override
    public DeltrDoc doc(int index) {
        return (DeltrDoc) this.scoreDocs[index];
    }
}
