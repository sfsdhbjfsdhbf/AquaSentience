package com.ywy.ywyagent.agent.model;

import java.util.*;

public class TfidfSimilarity {

    // 分词（简单基于空格，可替换更强 tokenizer）
    private List<String> tokenize(String text) {
        return Arrays.asList(text.toLowerCase().split("\\s+"));
    }

    // 计算 TF: term frequency
    private Map<String, Double> computeTF(List<String> tokens) {
        Map<String, Double> tf = new HashMap<>();
        for (String t : tokens) tf.put(t, tf.getOrDefault(t, 0.0) + 1.0);
        int size = tokens.size();
        tf.replaceAll((k, v) -> v / size);
        return tf;
    }

    // 计算 IDF: inverse document frequency
    private Map<String, Double> computeIDF(List<String> t1, List<String> t2) {
        Set<String> all = new HashSet<>();
        all.addAll(t1);
        all.addAll(t2);

        Map<String, Double> idf = new HashMap<>();
        for (String term : all) {
            int count = 0;
            if (t1.contains(term)) count++;
            if (t2.contains(term)) count++;

            idf.put(term, Math.log(2.0 / (count + 1))); // 平滑 IDF
        }
        return idf;
    }

    // 生成 TF-IDF 向量
    private Map<String, Double> computeTFIDF(List<String> tokens,
                                             Map<String, Double> idf) {

        Map<String, Double> tf = computeTF(tokens);
        Map<String, Double> tfidf = new HashMap<>();

        for (String term : idf.keySet()) {
            tfidf.put(term, tf.getOrDefault(term, 0.0) * idf.get(term));
        }
        return tfidf;
    }

    // 计算 Cosine 相似度
    private double cosine(Map<String, Double> v1, Map<String, Double> v2) {
        double dot = 0, normA = 0, normB = 0;

        for (String k : v1.keySet()) {
            double a = v1.get(k);
            double b = v2.get(k);
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
    }

    // 对外统一接口：计算文本相似度
    public double similarity(String s1, String s2) {
        List<String> t1 = tokenize(s1);
        List<String> t2 = tokenize(s2);

        Map<String, Double> idf = computeIDF(t1, t2);

        Map<String, Double> v1 = computeTFIDF(t1, idf);
        Map<String, Double> v2 = computeTFIDF(t2, idf);

        return cosine(v1, v2);
    }
}
