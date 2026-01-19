package com.ywy.ywyagent.tools;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PDFGenerationToolTest {

    @Test
    void generatePDF() {
        PDFGenerationTool tool = new PDFGenerationTool();
        String fileName = ".pdf";
        String result = tool.generatePDF(
                "test.pdf",
                "你好，老八！\n这是一个 PDF 测试内容。\n支持中文，也支持 English. \uD83D\uDE04 \uD83D\uDE00 \uD83D\uDE0A \uD83D\uDE09 \uD83D\uDE06 \uD83D\uDE02 \uD83E\uDD23 "
        );

        assertNotNull(result);

    }
}