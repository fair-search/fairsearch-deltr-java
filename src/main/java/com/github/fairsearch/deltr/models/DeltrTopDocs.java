package com.github.fairsearch.deltr.models;

import java.util.List;

/**
 * This is an interface describing the methods that a query/ranking must have so that DELTR would function
 */
public interface DeltrTopDocs {

    /**
     * A unique id of the query/ranking (if multiple queries are inputted)
     * @return
     */
    int id();

    /**
     * The number of documents in the ranking
     * @return
     */
    int size();

    /**
     * Returns a copy of the documents in the ranking
     * @return
     */
    List<DeltrDoc> docs();

    /**
     * Returns the document in the ranking at position `index`
     * @param index
     * @return
     */
    DeltrDoc doc(int index);
}
