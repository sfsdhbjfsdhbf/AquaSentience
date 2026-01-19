package com.ywy.ywyagent.controller;

import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * PDF文件下载控制器
 *
 * 前端调用：GET /api/ai/pdf/download?path=C:\Users\...\report.pdf
 */
@RestController
@RequestMapping("/ai/pdf")
public class PdfDownloadController {

    // PDF文件存储的基础路径（根据你的实际存储路径修改）
    // Windows路径示例
    private static final String PDF_BASE_PATH = "C:\\Users\\Administrator\\code\\java\\ywyagent\\tmp\\pdf\\";

    // 如果需要使用相对路径，可以这样设置：
    // private static final String PDF_BASE_PATH = "./tmp/pdf/";

    // Linux/Mac路径示例：
    // private static final String PDF_BASE_PATH = "/tmp/pdf/";

    /**
     * PDF文件下载接口
     *
     * 前端调用：GET /api/ai/pdf/download?path=C:\Users\...\report.pdf
     *
     * @param pdfPath PDF文件的完整路径
     * @return PDF文件流
     */
    @GetMapping("/download")
    public ResponseEntity<Resource> downloadPdf(@RequestParam("path") String pdfPath) {
        try {
            // 参数验证
            if (pdfPath == null || pdfPath.trim().isEmpty()) {
                return ResponseEntity.badRequest().build();
            }

            // 安全检查：确保路径在允许的目录下
            Path filePath = Paths.get(pdfPath);
            Path basePath = Paths.get(PDF_BASE_PATH).toAbsolutePath().normalize();
            Path normalizedPath = filePath.toAbsolutePath().normalize();

            // 验证路径是否在允许的目录下（防止路径遍历攻击）
            if (!normalizedPath.startsWith(basePath)) {
                return ResponseEntity.badRequest().build();
            }

            File file = normalizedPath.toFile();

            // 检查文件是否存在
            if (!file.exists() || !file.isFile()) {
                return ResponseEntity.notFound().build();
            }

            // 验证文件扩展名
            String fileName = file.getName();
            if (!fileName.toLowerCase().endsWith(".pdf")) {
                return ResponseEntity.badRequest().build();
            }

            // 创建Resource对象
            Resource resource = new FileSystemResource(file);

            // 设置响应头
            HttpHeaders headers = new HttpHeaders();

            // 设置Content-Disposition，指定下载文件名
            // 处理中文文件名编码
            String encodedFileName = java.net.URLEncoder.encode(fileName, "UTF-8")
                    .replaceAll("\\+", "%20");
            headers.add(HttpHeaders.CONTENT_DISPOSITION,
                    "attachment; filename=\"" + fileName + "\"; filename*=UTF-8''" + encodedFileName);

            // 设置Content-Type
            headers.add(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_PDF_VALUE);

            // 设置Content-Length
            headers.add(HttpHeaders.CONTENT_LENGTH, String.valueOf(file.length()));

            return ResponseEntity.ok()
                    .headers(headers)
                    .contentType(MediaType.APPLICATION_PDF)
                    .body(resource);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().build();
        }
    }
}
