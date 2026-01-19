package com.ywy.ywyagent.tools;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;

public class WebScrapingTooling {
    @Tool(description = "Scape the content of a web page")
    public String scapeWebPage(@ToolParam(description = "URL of the web page to scape" )String url){
        try {
            Document doc = Jsoup.connect(url).get();
            return doc.html();
        }catch (Exception e){
            return "Error scraping web page: " + e.getMessage();
        }

    }
}
