package com.ywy.ywyagent.tools;

import cn.hutool.core.io.FileUtil;
import com.itextpdf.kernel.font.PdfFont;
import com.itextpdf.kernel.font.PdfFontFactory;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;

import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Paragraph;
import com.ywy.ywyagent.service.PdfService;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;

@Component
public class PDFGenerationTool implements ApplicationContextAware {

    private static ApplicationContext applicationContext;
    private static PdfService pdfService;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        PDFGenerationTool.applicationContext = applicationContext;
        PDFGenerationTool.pdfService = applicationContext.getBean(PdfService.class);
    }

    private PdfService getPdfService() {
        if (pdfService != null) {
            return pdfService;
        }
        if (applicationContext != null) {
            return applicationContext.getBean(PdfService.class);
        }
        return null;
    }

    @Tool(description = "Generate a PDF file with given content")
    public String generatePDF(
            @ToolParam(description = "Name of the file to save the generated PDF") String fileName,
            @ToolParam(description = "Content to be included in the PDF") String content
    ){
        String fileDir = FileConstant.FILE_SAVE_DIR + "/pdf";
        String filePath = fileDir + "/" + fileName;
        try {
            FileUtil.mkdir(fileDir);

            try (PdfWriter writer = new PdfWriter(filePath);
                 PdfDocument pdf = new PdfDocument(writer);
                 Document document = new Document(pdf)) {

                // 关键：改成你自己的 NotoSansCJK 字体，放在 resources/fonts 里
                String fontPath = "src/main/resources/fonts/NotoSansCJKsc-Regular.otf";

                PdfFont pdfFont = PdfFontFactory.createFont(
                        fontPath,
                        com.itextpdf.io.font.PdfEncodings.IDENTITY_H
                );

                document.setFont(pdfFont);
                document.setFontSize(12);

                Paragraph paragraph = new Paragraph(content);
                document.add(paragraph);
            }
            PdfService service = getPdfService();
            if (service != null) {
                String chatId = getCurrentChatId(service);
                if (chatId != null) {
                    service.sendPdfGeneratedEvent(chatId, filePath);
                }
            }

            return "PDF generated successfully " ;

        } catch (Exception e) {
            e.printStackTrace();
            return "Error generating PDF: " + e.getMessage();
        }
    }

    private String getCurrentChatId(PdfService service) {
        return service.getCurrentChatId();
    }
}
