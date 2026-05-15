#!/usr/bin/env python3
"""Generate concrete-task version of demo_task_selector_fancy100.html"""
from __future__ import annotations

from pathlib import Path


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def fmt(t: dict) -> str:
    return (
        "      { "
        f'id:{t["id"]}, title:"{esc(t["title"])}", demo:"{esc(t["demo"])}", '
        f'difficulty:"{t["difficulty"]}", scene:"{esc(t["scene"])}", sourceName:"{esc(t["sourceName"])}", '
        f'sourceUrl:"{t["sourceUrl"]}", mediaUrl:"{t["mediaUrl"]}", mediaVerified:true '
        + "}"
    )


def main() -> None:
    T: list[dict] = []

    def add(
        i: int,
        title: str,
        demo: str,
        diff: str,
        scene: str,
        evidence: str,
        src: str,
        media: str,
    ) -> None:
        T.append(
            dict(
                id=i,
                title=title,
                demo=demo,
                difficulty=diff,
                scene=scene,
                sourceName=evidence,
                sourceUrl=src,
                mediaUrl=media,
            )
        )

    # --- 灵巧装配 (1–22)
    add(1, "双臂将柱形电池沿导向槽推入仓体直至止挡", "电池顶面与仓口齐平、无歪斜卡死", "困难", "灵巧装配", "ALOHA 论文演示", "https://tonyzhaozh.github.io/aloha", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(2, "双臂将扎带头穿过底板方孔再回穿并收紧塑条扣", "扎带扣锁死、尾部拉直无余摆", "困难", "灵巧装配", "ALOHA 论文演示", "https://tonyzhaozh.github.io/aloha", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(3, "双臂揭开杯口封膜并掀开酱料小杯盖", "杯口完全敞开可倒出酱料", "困难", "灵巧装配", "ALOHA 论文演示", "https://tonyzhaozh.github.io/aloha", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(4, "双臂将网线水晶头水平压入路由器口直到弹片锁住", "轻拉网线不脱出", "困难", "灵巧装配", "ALOHA 路由器组装节选", "https://arxiv.org/abs/2304.13705", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(5, "双臂将小电路板对齐另一板圆柱销按下直到贴合", "两板贴合无翘起缝隙", "困难", "灵巧装配", "ALOHA 组装节选", "https://arxiv.org/abs/2304.13705", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(6, "双臂将齿轮毂对准轴端推入并旋转微调啮合", "齿轮可回转、无径向松动", "困难", "灵巧装配", "ALOHA Unleashed", "https://aloha-unleashed.github.io/", "https://aloha-unleashed.github.io/")
    add(7, "双臂在给定运动鞋上系标准蝴蝶结鞋带", "双耳对称受力可跑动试拉", "困难", "灵巧装配", "ALOHA Unleashed", "https://aloha-unleashed.github.io/", "https://aloha-unleashed.github.io/")
    add(8, "双臂将圆领 T 恤下摆张开套入衣架横梁后抚平挂起", "肩缝不偏、下摆不拖地", "困难", "灵巧装配", "ALOHA Unleashed", "https://aloha-unleashed.github.io/", "https://aloha-unleashed.github.io/")
    add(9, "双臂捏住 USB‑C 插头两侧对准母座直推至触底", "需按键才能拔出", "中等", "灵巧装配", "双臂插拔技能", "https://tonyzhaozh.github.io/aloha", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(10, "双臂侧向把持销钉对准盲孔边滑边压直至全长没入", "末端无翘起", "困难", "灵巧装配", "仿真任务类比 PegInsertionSide", "https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html", "https://maniskill.readthedocs.io/en/latest/user_guide/demos/gallery.html")
    add(11, "双臂夹持三相插头对齐排插导向一次推到底", "插脚完全入孔、盖板回弹", "中等", "灵巧装配", "仿真 PlugCharger 类比", "http://maniskill2.github.io/", "http://maniskill2.github.io/")
    add(12, "一臂理直线缆另一臂将线卡扣压入底板卡座", "线束平直不拱起脱扣", "中等", "灵巧装配", "线缆整理", "https://tonyzhaozh.github.io/aloha", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(13, "双臂将小合页销自上而下敲入座板与门框铰链筒", "门扇开阖无剐蹭声", "困难", "灵巧装配", "家具装配节选", "https://clvrai.github.io/furniture-bench/", "https://clvrai.github.io/furniture-bench/")
    add(14, "双臂将木制凸榫对准另一板预埋孔垂直敲入榫肩贴齐", "摇动无榫松动感", "困难", "灵巧装配", "IKEA‑style FurnitureBench", "https://clvrai.github.io/furniture-bench/", "https://clvrai.github.io/furniture-bench/")
    add(15, "双臂将三色积木依次自下而上居中堆叠成三层塔", "俯视中心偏差小于半块厚度", "中等", "灵巧装配", "Isaac Cube Stack", "https://github.com/isaac-sim/IsaacLab", "https://github.com/isaac-sim/IsaacLab/discussions/1632")
    add(16, "双臂同时从两侧托底抬起同一纸箱离地约 5cm", "抬升平稳无碰撞两臂", "中等", "灵巧装配", "仿真双臂抬箱", "https://github.com/FAR3L/IsaacLab", "https://github.com/FAR3L/IsaacLab")
    add(17, "右臂把持积木递向左臂接物点后左臂精确放到坐标 pad", "落点落入 2cm 方格", "困难", "灵巧装配", "Franka Duo handover", "https://franka.de/fr3-duo", "https://franka.de/fr3-duo")
    add(18, "双臂将面包底片‑火腿芝士‑顶面包对齐后合上并轻压实", "切面整齐不外溢", "中等", "灵巧装配", "三明治装配线节选", "https://www.youtube.com/watch?v=qcdvrRyk8Kk", "https://www.youtube.com/watch?v=qcdvrRyk8Kk")
    add(19, "双臂将便当盒四角卡扣同时对压直至全部咔哒闭锁", "颠倒盒身无盖分离", "中等", "灵巧装配", "餐厨整理类比", "https://www.figure.ai/news/helix-02", "https://www.figure.ai/news/helix-02")
    add(20, "双臂拧开调味瓶倒出定量颗粒后拧紧螺纹盖", "无撒漏在桌面禁区", "中等", "灵巧装配", "流体倾倒 Diffusion Policy", "https://diffusion-policy.cs.columbia.edu/", "https://diffusion-policy.cs.columbia.edu/")
    add(21, "双臂将口朝下的马克杯翻正并使把手朝向桌缘标记", "杯底落下无声", "中等", "灵巧装配", "杯子翻转 rollout", "https://arxiv.org/abs/2303.04137", "https://diffusion-policy.cs.columbia.edu/")
    add(22, "双臂用刮刀将碗内酱料耙平涂于方形吐司单面", "面包区外无污染", "困难", "灵巧装配", "涂抹 rollout", "https://diffusion-policy.cs.columbia.edu/", "https://diffusion-policy.cs.columbia.edu/")

    # --- 软性物体 (23–34)
    add(23, "双臂对角抖平毛巾后对折两次成小矩形块", "四边线与桌缘平行", "困难", "软性物体", "Figure Helix Laundry", "https://www.figure.ai/news/helix-learns-to-fold-laundry", "https://www.figure.ai/news/helix-learns-to-fold-laundry")
    add(24, "双臂抓取揉皱 T 恤两腋下向两侧牵拉直至肩线摆正", "主轴线平直无扭转褶皱环", "困难", "软性物体", "Cloth Funnels TRI", "https://clothfunnels.cs.columbia.edu/", "https://clothfunnels.cs.columbia.edu/")
    add(25, "双臂夹住抱枕套拉链拉头从一端匀速滑至另一端闭口", "链齿无缺齿脱轨", "中等", "软性物体", "可变形 TRI DextAIRity", "http://www.tri.global/research/dextairity-deformable-manipulation-can-be-breeze", "http://www.tri.global/research/dextairity-deformable-manipulation-can-be-breeze")
    add(26, "双臂将 2m 网线绕成双环再用魔术贴在中间束紧", "垂落单端长度小于 30cm", "简单", "软性物体", "理线技能", "https://tonyzhaozh.github.io/aloha", "https://www.youtube.com/watch?v=VUxFhtGWD7w")
    add(27, "双臂四角提起防尘布盖住 30cm 方台布边垂落等高", "布面中心无破洞褶皱峰", "中等", "软性物体", "桌面铺布", "https://www.physicalintelligence.company/blog/pi0", "https://www.physicalintelligence.company/blog/pi0")
    add(28, "双臂撑开 Laundry 绳袋开口塞入三件团衣并抽绳打活结", "倒置袋口不落衣", "困难", "软性物体", "衣物收纳类比", "https://www.physicalintelligence.company/blog/openpi", "https://www.physicalintelligence.company/blog/openpi")
    add(29, "双臂抓取圆形蒸笼无纺布对角铺入围笼底边贴壁", "底面无直径大于 5cm 空洞", "中等", "软性物体", "炊事布景 Mobile ALOHA", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(30, "双臂翻出棉袜内里后由袜尖卷至袜口包住成旅行卷", "卷体直径近似均匀", "中等", "软性物体", "内务整理类比", "https://www.figure.ai/news/helix-02", "https://www.figure.ai/news/helix-02")
    add(31, "双臂提拉垃圾袋上口外翻套在矩形桶四角卡边固定", "外沿均匀无单侧滑落", "简单", "软性物体", "TidyBot", "https://tidybot.cs.princeton.edu/", "https://tidybot.cs.princeton.edu/")
    add(32, "双臂对角抽离铺在圆桌上的大餐巾使餐具原位不倒", "三套餐具直立无翻倒", "困难", "软性物体", "Cloth funnel 高级", "https://clothfunnels.cs.columbia.edu/", "https://clothfunnels.cs.columbia.edu/")
    add(33, "双臂浴巾长对折再对折后紧密卷筒", "卷缝呈直线不外鼓", "中等", "软性物体", "Optimus 折衣节选", "https://www.youtube.com/shorts/8vsTNFUFJEU", "https://www.youtube.com/shorts/8vsTNFUFJEU")
    add(34, "双臂握住衣架钩子将衬衫下摆完全脱离衣架后摊平台面", "前襟中线与台面纵轴对齐", "中等", "软性物体", "ALOHA Unleashed 逆操作", "https://aloha-unleashed.github.io/", "https://aloha-unleashed.github.io/")

    # --- 堆叠分拣 (35–46)
    add(35, "按颜色分拣混放正方体积木到红黄蓝三托盘", "每托盘零异色混入", "中等", "堆叠分拣", "Phoenix 桌面分拣节选", "https://www.youtube.com/watch?v=9AofrJJaS8U", "https://www.youtube.com/watch?v=9AofrJJaS8U")
    add(36, "将汤匙叉刀各放入对应竖槽分类盒", "三槽各司其职", "中等", "堆叠分拣", "餐具分拣节选", "https://www.figure.ai/news/helix-02", "https://www.figure.ai/news/helix-02")
    add(37, "将散装零食直立袋单层排满托盘前沿标签齐平 Retail fronting", "前沿线偏差小于 5mm", "中等", "堆叠分拣", "零售理货", "https://english.kyodonews.net/articles/-/60709", "https://english.kyodonews.net/articles/-/60709")
    add(38, "将混入的螺丝刀与记号笔分拣到左右托盘", "无交叉污染", "中等", "堆叠分拣", "RH20T 示教范式", "https://rh20t.github.io/", "https://rh20t.github.io/")
    add(39, "双侧推拨金属滚球落入桌面凹槽夹具凹槽中心", "球静止不弹跳出轨", "中等", "堆叠分拣", "RollBall ManiSkill", "http://maniskill2.github.io/", "http://maniskill2.github.io/")
    add(40, "三套不同口径杯子从大到小同轴套嵌", "小杯最低点触大杯底", "困难", "堆叠分拣", "堆叠语义技能", "https://say-can.github.io/", "https://say-can.github.io/")
    add(41, "夹持纸盒旋转直至条码朝上对准虚拟扫码基准线", "条码长边与射线夹角小于 10°", "简单", "堆叠分拣", "扫码台流程", "https://english.kyodonews.net/articles/-/60709", "https://english.kyodonews.net/articles/-/60709")
    add(42, "将一个空塑料瓶与瓶盖分别投进左右圆孔回收治具", "两个物体均未卡在孔沿", "简单", "堆叠分拣", "双臂分类节拍", "https://maniskill.readthedocs.io/en/latest/tasks/", "https://maniskill.readthedocs.io/en/latest/tasks/")
    add(43, "从垛叠纸杯顶仅分离单杯放于杯架环", "下杯不因摩擦脱离垛", "中等", "堆叠分拣", "咖啡站节流", "https://www.artlybaristabot.com/demo", "https://www.artlybaristabot.com/demo")
    add(44, "单手稳定蛋托另一手取单枚鸡蛋平移置入硅胶煮蛋圈", "壳无裂痕漏液", "困难", "堆叠分拣", "易碎抓取 DROID spirit", "https://droid-dataset.github.io/", "https://droid-dataset.github.io/")
    add(45, "从便签本撕单张按压力贴于磁吸背板矩形框内", "四边均被框线覆盖无超出", "简单", "堆叠分拣", "CLIPort‑style pick‑place", "https://cliport.github.io/", "https://cliport.github.io/")
    add(46, "将混放金属圆片按蚀刻面值分入四宫格托盘", "每格仅单色面值", "中等", "堆叠分拣", "YuMi 小件节拍", "https://www.youtube.com/watch?v=ArBxq3mOt2s", "https://www.youtube.com/watch?v=ArBxq3mOt2s")

    # --- 门柜流体 (47–56)
    add(47, "双臂全开对开矮柜各一扇门后将炖锅端到台面再关门", "两扇门磁吸闭合无缝", "困难", "门柜流体", "Mobile ALOHA 橱柜剪辑", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(48, "拉开单层抽屉拉出止挡取出马克杯再推回扣手", "抽屉面板与柜体Flush", "中等", "门柜流体", "RoboCasa drawer", "https://robocasa.ai/", "https://robocasa.ai/")
    add(49, "旋转冷水龙头旋钮过水 3s 后回转关闭水流停止", "管口无水线滴落 5s", "中等", "门柜流体", "冲洗锅剪辑", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(50, "左手扶锅右手模拟拧动燃气灶旋钮一格", "模拟火焰指示刻度对齐标记", "中等", "门柜流体", "厨房旋钮", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(51, "下拉打开微波炉门将饭盒置于转盘中心再关门按启动", "门缝感测不到阻挡", "困难", "门柜流体", "Kitchen 链路", "https://robocasa.ai/", "https://robocasa.ai/")
    add(52, "横向拉开冰箱门下层层架取矿泉水再推回门关严", "门封条负压感", "中等", "门柜流体", "便利店冷柜", "https://robocasa.ai/", "https://robocasa.ai/")
    add(53, "先解除饮水机童锁长按再点热水键接八成满纸杯", "杯身无挤压变形喷水", "中等", "门柜流体", "安全互锁类比", "https://say-can.github.io/", "https://say-can.github.io/")
    add(54, "双手抓窗扣侧向推门扇至终点再拉回闭锁", "窗扇无脱轨咔嚓声", "简单", "门柜流体", "Meta‑World slide", "https://meta-world.github.io/", "https://meta-world.github.io/")
    add(55, "三序拨轮转密码盘至公开组合线开微型保险柜门板", "门弹开角大于 45°", "困难", "门柜流体", "旋钮组合", "https://meta-world.github.io/", "https://meta-world.github.io/")
    add(56, "下拉烤箱门拉出烤网托盘再推回笼合门拉手", "门磁吸闭锁", "困难", "门柜流体", "双层 oven RLBench‑style", "https://github.com/stepjam/RLBench", "https://www.youtube.com/watch?v=bKaK_9O3v7Y")

    # --- 烹调食物 (57–62)
    add(57, "双臂一锅一铲爆炒虾仁直至虾体均匀变色", "无虾跳出锅缘", "困难", "烹调食物", "Mobile ALOHA 虾仁剪辑", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(58, "右倾热锅左用勺子将虾仁拨入碗内限位圆圈", "汤汁不溢圈外超过 30%", "中等", "烹调食物", "装盘", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(59, "双手撕开碗面封口倒入沸水至刻度线盖住压盖", "30s 无大量蒸汽侧喷", "中等", "烹调食物", "BEHAVIOR 厨房节选", "https://behavior.stanford.edu/", "https://behavior.stanford.edu/")
    add(60, "持长筷搅动锅内垂直下入的面束防粘底 20s", "面条无块状黏连", "困难", "烹调食物", "煮面链路", "https://behavior.stanford.edu/", "https://behavior.stanford.edu/")
    add(61, "用夹子从仿汤锅网格取鱼丸放入纸质汤碗三只", "每碗等量三只", "中等", "烹调食物", "熟食夹取", "https://www.youtube.com/watch?v=9AofrJJaS8U", "https://www.youtube.com/watch?v=9AofrJJaS8U")
    add(62, "将吐司火腿芝士三明治对角切成两个等腰直角三角形", "刀口干净利落无馅挤出", "中等", "烹调食物", "食品加工线节选", "https://www.youtube.com/watch?v=qcdvrRyk8Kk", "https://www.youtube.com/watch?v=qcdvrRyk8Kk")

    # --- 人形搬运 (63–74)
    add(63, "双手取马克杯轻放入洗碗机顶层杯架凹槽", "杯耳不妨碍喷淋臂虚拟包络", "困难", "人形搬运", "Figure Helix 02 洗碗", "https://www.figure.ai/news/helix-02", "https://www.figure.ai/news/helix-02")
    add(64, "双手从最下层碗架端起整叠餐盘上移到台面置物架第二层", "盘塔无倾斜滑动", "中等", "人形搬运", "Helix02 餐厨", "https://www.figure.ai/news/helix-02", "https://www.figure.ai/news/helix-02")
    add(65, "左右手交接塑料杯后放入洗碗机下层格", "转手区无掉落抖动", "困难", "人形搬运", "1X GTC Dishwasher", "https://www.1x.tech/discover/1X-NVIDIA-Research-Collaboration", "https://www.youtube.com/watch?v=bUrLuUxv9gE")
    add(66, "双手捧起发动机塑料盖板从一料架平移到另一料架就位", "全程水平倾角小于 15°", "困难", "人形搬运", "Boston Dynamics Atlas 搬运", "https://www.youtube.com/watch?v=F_7IPm7f1vI", "https://www.youtube.com/watch?v=F_7IPm7f1vI")
    add(67, "盖板在胸口高度换手翻转 180° 后放回指定销定位", "销孔一次对准无反复搜索", "困难", "人形搬运", "Atlas‑style regrasp", "https://bostondynamics.com/atlas/", "https://www.youtube.com/watch?v=F_7IPm7f1vI")
    add(68, "双臂从滚筒末端抱下瓦楞箱码到托盘左上角对齐边条", "箱角与托盘角视觉重合", "中等", "人形搬运", "Apptronik 搬箱节选", "https://apptronik.com/apollo", "https://apptronik.com/videos")
    add(69, "从纸箱取瓶装饮料填入冷柜滑道空缺第一排贴冷墙", "瓶标正面朝外与高瓶脊线对齐", "中等", "人形搬运", "Telexistence Gordon", "https://tx-inc.com/en/blog/2021/11/02/11451/", "https://tx-inc.com/en/blog/2021/11/02/11451/")
    add(70, "抱起 5 kg 砝码箱行走三步轻放地坪方框内无弹跳", "落点脚掌未触框外边", "困难", "人形搬运", "Digit 重物搬运节选", "https://www.agilityrobotics.com/videos/precision-in-motion-five-humanoid-material-handling-workflows", "https://www.agilityrobotics.com/videos/precision-in-motion-five-humanoid-material-handling-workflows")
    add(71, "在周转箱内将高矮瓶罐重排前排最矮后背最高", "三列目测单调", "中等", "人形搬运", "Digit 理货节选", "https://www.agilityrobotics.com/", "https://www.agilityrobotics.com/videos/timelapse-26-hours-of-humanoid-robots-working-autonomously")
    add(72, "俯身双手捡起地面条状工具起立插入腰带挂环扣", "身体不碰倒旁侧椅脚", "困难", "人形搬运", "电动 Atlas teaser", "https://www.youtube.com/watch?v=raYWbqbZbmc", "https://www.youtube.com/watch?v=raYWbqbZbmc")
    add(73, "视觉遮罩下手触寻找并一次闭合抓住桌上圆柱摆件", "首次闭合离地高度不改变基座位移", "困难", "人形搬运", "Sanctuary tactile", "https://www.youtube.com/watch?v=E4RqGYbxaWM", "https://www.youtube.com/watch?v=E4RqGYbxaWM")
    add(74, "在胸前托盘微调纸杯 Logo 转角直到正对虚拟顾客相机", "偏航误差小于 ±7°", "简单", "人形搬运", "咖啡递送节选", "https://www.artlybaristabot.com/", "https://www.artlybaristabot.com/demo")

    # --- 零售补给 (75–82)
    add(75, "开箱取出四连装饮料并排压入冷柜重力滑道卡条", "滑道满陈列无悬空卡瓶", "中等", "零售补给", "便利店补货节选", "https://english.kyodonews.net/articles/-/60709", "https://english.kyodonews.net/articles/-/60709")
    add(76, "将纸质价签上部插入货架前横梁透明夹缝中推到底", "价签readable 无外翘", "简单", "零售补给", "价签 insertion", "https://tx-inc.com/en/blog/2022/08/10/11712/", "https://tx-inc.com/en/blog/2022/08/10/11712/")
    add(77, "开小锁橱窗取一包指定 SKU 放回后锁舌咔哒闭锁", "门缝均匀", "中等", "零售补给", "Lawson 货架实验类比", "https://japanfoodnews.com/153503/2026/01/21/distribution-and-food-service/retail/cvs/", "https://japanfoodnews.com/153503/2026/01/21/distribution-and-food-service/retail/cvs/")
    add(78, "顾客篮中最重长颈瓶右臂递左臂后放到扫码台凹槽", "瓶底完全入槽不靠边倒", "中等", "零售补给", "双臂 handover + 称重联想", "https://franka.de/fr3-duo", "https://franka.de/fr3-duo")
    add(79, "将开箱饭团阵列压入展示托盘每格凹槽防侧旋", "12 格里无翘起 >15°", "中等", "零售补给", "鲜食陈列", "https://www.figure.ai/news/helix-02", "https://www.figure.ai/news/helix-02")
    add(80, "撕开连包吸管小口抽单支插入注塑杯盖十字孔直至肩台", "杯盖不因力翘裂", "简单", "零售补给", "配套动作", "https://www.artlybaristabot.com/demo", "https://www.artlybaristabot.com/demo")
    add(81, "将背胶买一送一拍平粘附于立式货架立柱 120cm 标高区", "无气泡中空直径超 15mm", "中等", "零售补给", "门店物料", "https://tx-inc.com/en/blog/2022/08/10/11712/", "https://tx-inc.com/en/blog/2022/08/10/11712/")
    add(82, "从最上层挑出贴红标过期汽水罐置入红色退货箱开口", "无掉落罐在箱沿外", "简单", "零售补给", "逆向物流", "https://english.kyodonews.net/articles/-/60709", "https://english.kyodonews.net/articles/-/60709")

    # --- 语言 / 语义条件任务 (具象) (83–92)
    add(83, "听到指令后仅从桌上玩具中抓起恐龙放回空篮其余不动", "非恐龙玩具零位移", "中等", "语言条件", "RT‑2 extinct animal demo", "https://robotics-transformer2.github.io/", "https://robotics-transformer2.github.io/")
    add(84, "听到「无糖」形容词后从三罐饮料中取贴无糖标那罐移到绿盘", "有糖饮料罐位置不变", "困难", "语言条件", "Open‑vocab grasp 类比 RT‑2", "https://robotics-transformer2.github.io/", "https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/")
    add(85, "听到中英混合句「把最长的积木放到白板右侧」完成任务", "仅最长条发生位移终点正确", "困难", "语言条件", "VLA rollout 抽象", "https://openvla.github.io/", "https://openvla.github.io/")
    add(86, "随语音步骤链：开抽屉‑取刷子‑关抽屉‑放到水槽旁", "每步语义触发间无提前执行", "困难", "语言条件", "CALVIN 长链条", "https://calvin.informatik.uni-freiburg.de/", "http://calvin.informatik.uni-freiburg.de/")
    add(87, "见图片目标框提示将红色立方体叠在蓝色立方体上", "色块判别正确容错照明变化", "中等", "语言条件", "Goal image / Octo‑style", "https://octo-models.github.io/", "https://octo-models.github.io/")
    add(88, "读秒表式指令倒计时内把散钉全部扫入金属簸箕", "桌面剩余钉数为 0", "中等", "语言条件", "Time‑budget policy", "https://innermonologue.github.io/", "https://innermonologue.github.io/")
    add(89, "当语音说「如果没绿色块就退出」场景中无绿色时机械臂归零", "无多余 grasp 尝试", "困难", "语言条件", "条件分支 LM + robot", "https://code-as-policies.github.io/", "https://code-as-policies.github.io/")
    add(90, "复述人类纠正句后撤回错误放置的杯子放回起点", "一次纠正内完成", "困难", "语言条件", "纠错对话规划", "https://palm-e.github.io/", "https://palm-e.github.io/")
    add(91, "多模态 Prompt 含 Emoji 所指水果放入对应象形碗", "三水果零混放", "中等", "语言条件", "VIMA bench", "https://vimalabs.github.io/", "https://vimalabs.github.io/")
    add(92, "同时接收语音和小地图图标将夹子送到「星标」工件格", "落点欧式距离小于一格宽", "困难", "语言条件", "多模态 embodied", "https://arxiv.org/abs/2304.11752", "https://arxiv.org/abs/2304.11752")

    # --- 移动 / whole‑home （缩比或桌面等价） (93–96)
    add(93, "推移动底座到水槽位后双手抓锅伸向龙头接水阈值线", "停准误差轮印不越警戒线", "困难", "移动基底", "Mobile ALOHA 导航+操作剪辑", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(94, "推基地到 Elevator Button 台前抬手按上行键再后退等待", "键帽触点 LED 亮起", "困难", "移动基底", "电梯呼叫剪辑", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(95, "移动中双手各持拖把杆两端擦除地面矩形渍迹", "RGB 阈值内污渍面积下降 95%", "困难", "移动基底", "擦地剪辑 Mobile ALOHA", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")
    add(96, "从客厅点位导航到厨房的途中不停机抓取拖鞋放入侧篮", "拖鞋完全入篮不脱鞋跟挂边", "困难", "移动基底", "移动 manip 节选", "https://mobile-aloha.github.io/", "https://www.youtube.com/watch?v=zMNumQ45pJ8")

    # --- 数据采集 / 仿真技能（仍具象） (97–100)
    add(97, "用示教手柄记录一次「螺钉拧入六角孔」再给策略零样本回放", "最终扭矩超过阈值且无滑牙", "困难", "数据采集", "UmI 类比", "https://umi-gripper.github.io/", "https://umi-gripper.github.io/")
    add(98, "戴外骨骼单手执行复杂扭瓶盖动作映射到仿生手 replay", "开盖角位移连续无跳变", "困难", "数据采集", "DexCap", "https://dex-cap.github.io/", "https://dex-cap.github.io/")
    add(99, "在程序化生成kitchen中抓取随机材质马克杯放入洗碗机（sim）", "物理引擎无穿模卡住", "中等", "仿真生成", "GenSim/RoboGen pipeline", "https://robogen-ai.github.io/", "https://robogen-ai.github.io/")
    add(100, "在对抗视觉扰动下完成堆三块积木仿真 episode 不退化", "三堆叠成功后相机眩光贴片随机仍成功", "困难", "仿真鲁棒", "RoboArena", "https://robo-arena.github.io/", "https://robo-arena.github.io/")

    assert len(T) == 100

    js_lines = []
    for k, task in enumerate(T):
        line = fmt(task) + ("," if k < len(T) - 1 else "")
        js_lines.append(line)

    fancy_path = Path(__file__).resolve().parent / "demo_task_selector_fancy100.html"
    text = fancy_path.read_text(encoding="utf-8")
    start = text.index("const tasks = [") + len("const tasks = [")
    end = text.index("];", start)
    before = text[:start]
    after = text[end:]
    new_block = "\n" + "\n".join(js_lines) + "\n    "
    fancy_path.write_text(before + new_block + after, encoding="utf-8")

    # Update header subtitle + scene dropdown + KEY
    text = fancy_path.read_text(encoding="utf-8")
    text = text.replace(
        '<h1 class="title">Fancy Demo 任务池 · 100 条（论文顶会 + 公司产品 + 便利店可迁移）</h1>',
        '<h1 class="title">具体任务 Demo 清单 · 100 条（动作+物体+终点，可逐条勾选）</h1>',
    )
    text = text.replace(
        '<p class="sub">每条对应权威项目页/报道或官方影片；偏「展示冲击力强」的模仿学习与通才操作。请用本地 HTTP 打开以便嵌入 YouTube。</p>',
        '<p class="sub">每条先写<strong>能做的具体事</strong>；链接仅作<strong>同类任务佐证</strong>（原视频不一定是双臂 Franka）。场景 = 技能类型筛选。</p>',
    )
    text = text.replace(
        """        <select id="scene">
          <option value="all">来源：全部</option>
          <option value="论文顶会">论文顶会</option>
          <option value="公司产品">公司产品</option>
          <option value="可迁移桌面">可迁移桌面</option>
          <option value="公司产品+论文">公司产品+论文</option>
        </select>""",
        """        <select id="scene">
          <option value="all">场景：全部技能类型</option>
          <option value="灵巧装配">灵巧装配</option>
          <option value="软性物体">软性物体</option>
          <option value="堆叠分拣">堆叠分拣</option>
          <option value="门柜流体">门柜/流体旋钮</option>
          <option value="烹调食物">烹调/食物接触</option>
          <option value="人形搬运">人形/重物搬运可比</option>
          <option value="零售补给">零售/便利店补给</option>
          <option value="语言条件">语音或语义指令</option>
          <option value="移动基底">移动底座+上肢</option>
          <option value="数据采集">示教硬件/数据采集</option>
          <option value="仿真生成">程序化/生成式仿真</option>
          <option value="仿真鲁棒">对抗/鲁棒评测</option>
        </select>""",
    )
    text = text.replace('const KEY = "franka_demo_fancy100_v1"', 'const KEY = "franka_demo_fancy100_concrete_v2"')
    text = text.replace(
        '<input id="q" type="text" placeholder="搜索：ALOHA、Figure、OpenVLA、便利店…" />',
        '<input id="q" type="text" placeholder="搜索：插入、折叠、洗碗机、条形码、 язык…" />',
    )
    text = text.replace(
        '<a href="${t.sourceUrl}" target="_blank" rel="noreferrer">论文/新闻/项目（权威来源）</a>',
        '<a href="${t.sourceUrl}" target="_blank" rel="noreferrer">同类证据（论文页/官网/剪辑）</a>',
    )
    fancy_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
    print("OK")
