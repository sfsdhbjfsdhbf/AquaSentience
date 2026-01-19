package com.ywy.ywyagent.tools;

import cn.hutool.core.io.FileUtil;
import cn.hutool.http.HttpUtil;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;

import java.io.File;

@Component
public class ResourceDownloadTool implements ApplicationContextAware {

    private static ApplicationContext applicationContext;
    private static com.ywy.ywyagent.service.ResourceService resourceService;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        ResourceDownloadTool.applicationContext = applicationContext;
        // 延迟获取 ResourceService，确保它已经被初始化
        try {
            ResourceDownloadTool.resourceService = applicationContext.getBean(com.ywy.ywyagent.service.ResourceService.class);
        } catch (Exception e) {
            System.err.println("无法获取 ResourceService: " + e.getMessage());
        }
    }

    /**
     * 获取 ResourceService 实例
     * 如果直接注入失败，则从 ApplicationContext 获取
     */
    private com.ywy.ywyagent.service.ResourceService getResourceService() {
        // 方式1：如果静态变量已设置，直接使用
        if (resourceService != null) {
            return resourceService;
        }

        // 方式2：从 ApplicationContext 获取
        if (applicationContext != null) {
            try {
                return applicationContext.getBean(com.ywy.ywyagent.service.ResourceService.class);
            } catch (Exception e) {
                System.err.println("无法获取 ResourceService: " + e.getMessage());
                return null;
            }
        }

        return null;
    }

    @Tool(description = "download a resource from a given url")
    public String downloadResource(
            @ToolParam(description = "Url for the resource to download") String url,
            @ToolParam(description = "Name of the file to save the downloaded resource") String fileName) {
        String fileDir = FileConstant.FILE_SAVE_DIR + "/download";
        String filePath = fileDir + "/" + fileName;

        try {
            FileUtil.mkdir(fileDir);
            HttpUtil.downloadFile(url, new File(filePath));

            // ✅ 资源下载成功后，发送SSE事件通知前端
            com.ywy.ywyagent.service.ResourceService service = getResourceService();
            if (service != null) {
                String chatId = getCurrentChatId(service);
                if (chatId != null) {
                    service.sendResourceDownloadedEvent(chatId, filePath, fileName);
                } else {
                    System.out.println("警告：无法获取 chatId，资源下载事件未发送");
                }
            } else {
                System.out.println("警告：无法获取 ResourceService，资源下载事件未发送");
            }

            return "Resource downloaded successfully";
        } catch (Exception e) {
            e.printStackTrace();
            return "Error downloading resource: " + e.getMessage();
        }
    }

    /**
     * 获取当前chatId
     */
    private String getCurrentChatId(com.ywy.ywyagent.service.ResourceService service) {
        // 方式1：如果项目中有ChatIdContext（ThreadLocal），使用它
        // try {
        //     return ChatIdContext.get();
        // } catch (Exception e) {
        //     // 忽略
        // }

        // 方式2：从ResourceService获取（如果只有一个活跃连接）
        if (service != null) {
            try {
                return service.getCurrentChatId();
            } catch (Exception e) {
                System.err.println("获取 chatId 失败: " + e.getMessage());
            }
        }

        return null;
    }
}
