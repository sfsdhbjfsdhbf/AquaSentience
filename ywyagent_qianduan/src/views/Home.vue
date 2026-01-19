<template>
  <div class="home">
    <div class="home-container">
      <h1 class="home-title">欢迎来到AI应用中心</h1>
      <p class="home-description">选择下方的应用开始使用</p>
      <div class="app-grid">
        <router-link to="/manus" class="app-card">
          <div class="app-icon">🤖</div>
          <h3 class="app-name">AI超级智能体</h3>
          <p class="app-desc">强大的AI助手，为您提供智能对话服务</p>
        </router-link>
        <router-link to="/love" class="app-card">
          <div class="app-icon">🖼️</div>
          <h3 class="app-name">AI图像水印专家</h3>
          <p class="app-desc">专业的图像水印处理工具</p>
        </router-link>
        <router-link to="/full-image-protection" class="app-card">
          <div class="app-icon">🛡️</div>
          <h3 class="app-name">整图内容保护</h3>
          <p class="app-desc">上传图片并添加水印，保护您的图片内容</p>
        </router-link>
        <router-link to="/partial-content-protection" class="app-card">
          <div class="app-icon">🧩</div>
          <h3 class="app-name">局部内容保护</h3>
          <p class="app-desc">上传图片和掩码，添加局部水印保护</p>
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
// Home page component
</script>

<style scoped>
.home {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 40px 20px;
}

.home-container {
  max-width: 1200px;
  margin: 0 auto;
}

.home-title {
  font-size: 36px;
  font-weight: 700;
  color: #333;
  text-align: center;
  margin-bottom: 10px;
}

.home-description {
  font-size: 18px;
  color: #666;
  text-align: center;
  margin-bottom: 40px;
}

.app-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 30px;
  margin-top: 40px;
}

.app-card {
  background: white;
  border-radius: 16px;
  padding: 30px;
  text-decoration: none;
  color: inherit;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.app-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
}

.app-icon {
  font-size: 64px;
  margin-bottom: 20px;
}

.app-name {
  font-size: 24px;
  font-weight: 600;
  color: #333;
  margin-bottom: 10px;
}

.app-desc {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
}

@media (max-width: 768px) {
  .home-title {
    font-size: 28px;
  }

  .home-description {
    font-size: 16px;
  }

  .app-grid {
    grid-template-columns: 1fr;
    gap: 20px;
  }

  .app-card {
    padding: 25px;
  }
}
</style>
