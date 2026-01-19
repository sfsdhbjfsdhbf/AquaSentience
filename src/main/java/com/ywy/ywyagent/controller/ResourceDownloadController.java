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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URLEncoder;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 资源文件下载控制器
 *
 * 前端调用：GET /api/ai/resource/download?path=C:\Users\...\file.jpg
 */
@RestController
@RequestMapping("/ai/resource")
public class ResourceDownloadController {

    private static final Logger log = LoggerFactory.getLogger(ResourceDownloadController.class);

    // 资源文件存储的基础路径（根据你的实际存储路径修改）
    // Windows路径示例
    private static final String RESOURCE_BASE_PATH = "C:\\Users\\Administrator\\code\\java\\ywyagent\\tmp\\download\\";

    // 如果需要使用相对路径，可以这样设置：
    // private static final String RESOURCE_BASE_PATH = "./tmp/download/";

    // Linux/Mac路径示例：
    // private static final String RESOURCE_BASE_PATH = "/tmp/download/";

    /**
     * 资源文件下载接口
     *
     * 前端调用：GET /api/ai/resource/download?path=C:\Users\...\file.jpg
     *
     * @param resourcePath 资源文件的完整路径
     * @return 资源文件流
     */
    @GetMapping("/download")
    public ResponseEntity<Resource> downloadResource(@RequestParam("path") String resourcePath) {
        try {
            log.info("收到资源下载请求，路径: {}", resourcePath);

            // 参数验证
            if (resourcePath == null || resourcePath.trim().isEmpty()) {
                log.warn("资源路径参数为空");
                return ResponseEntity.badRequest().build();
            }

            // 安全检查：确保路径在允许的目录下
            Path filePath = Paths.get(resourcePath);
            Path basePath = Paths.get(RESOURCE_BASE_PATH).toAbsolutePath().normalize();
            Path normalizedPath = filePath.toAbsolutePath().normalize();

            // 验证路径是否在允许的目录下（防止路径遍历攻击）
            if (!normalizedPath.startsWith(basePath)) {
                log.warn("资源路径不在允许的目录下，请求路径: {}, 允许路径: {}", normalizedPath, basePath);
                return ResponseEntity.badRequest().build();
            }

            File file = normalizedPath.toFile();

            // 检查文件是否存在
            if (!file.exists() || !file.isFile()) {
                log.warn("资源文件不存在: {}", normalizedPath);
                return ResponseEntity.notFound().build();
            }

            String fileName = file.getName();
            log.info("开始下载资源文件: {}, 大小: {} bytes", fileName, file.length());

            // 创建Resource对象
            Resource resource = new FileSystemResource(file);

            // 根据文件扩展名确定Content-Type
            String contentType = getContentType(fileName);

            // 设置响应头
            HttpHeaders headers = new HttpHeaders();

            // 设置Content-Disposition，指定下载文件名
            // 处理中文文件名编码
            String encodedFileName = URLEncoder.encode(fileName, "UTF-8")
                    .replaceAll("\\+", "%20");
            headers.add(HttpHeaders.CONTENT_DISPOSITION,
                    "attachment; filename=\"" + fileName + "\"; filename*=UTF-8''" + encodedFileName);

            // 设置Content-Type
            headers.add(HttpHeaders.CONTENT_TYPE, contentType);

            // 设置Content-Length
            headers.add(HttpHeaders.CONTENT_LENGTH, String.valueOf(file.length()));

            return ResponseEntity.ok()
                    .headers(headers)
                    .contentType(MediaType.parseMediaType(contentType))
                    .body(resource);

        } catch (Exception e) {
            log.error("下载资源文件时发生错误", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * 根据文件扩展名获取Content-Type
     */
    private String getContentType(String fileName) {
        String lowerFileName = fileName.toLowerCase();
        if (lowerFileName.endsWith(".jpg") || lowerFileName.endsWith(".jpeg")) {
            return "image/jpeg";
        } else if (lowerFileName.endsWith(".png")) {
            return "image/png";
        } else if (lowerFileName.endsWith(".gif")) {
            return "image/gif";
        } else if (lowerFileName.endsWith(".webp")) {
            return "image/webp";
        } else if (lowerFileName.endsWith(".pdf")) {
            return "application/pdf";
        } else if (lowerFileName.endsWith(".zip")) {
            return "application/zip";
        } else {
            return "application/octet-stream";
        }
    }
}

