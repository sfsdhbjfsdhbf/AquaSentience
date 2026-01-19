package com.ywy.ywyagent.tools;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;
@SpringBootTest
class WebScrapingToolingTest {

    @Test
    void scapeWebPage() {
        WebScrapingTooling webScrapingTooling = new WebScrapingTooling();
        String url = "https://www.codefather.cn";
        String result = webScrapingTooling.scapeWebPage(url);
        assertNotNull(result);
    }
}