#include <cmath>
#include <memory>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QElapsedTimer>
#include <QUuid>
#include <QIcon>
#include <OgreCamera.h>
#include <OgreRay.h>
#include <OgrePlane.h>
#include <rviz_common/tool.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/view_controller.hpp>
#include <rviz_common/viewport_mouse_event.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_rendering/objects/billboard_line.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

namespace rviz_cloud_editor {
using String=std_msgs::msg::String;
static String message(const QJsonObject & o) {String m;m.data=QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();return m;}
class EraserTool:public rviz_common::Tool {
public:
 EraserTool(){shortcut_key_='e';}
 void onInitialize() override {
  setName("点云橡皮擦");setIcon(QIcon::fromTheme("draw-eraser"));
  auto node=context_->getRosNodeAbstraction().lock()->get_raw_node();
  pub_=node->create_publisher<String>("/map_editor/command",20);
  settings_=node->create_subscription<String>("/map_editor/settings",rclcpp::QoS(1).transient_local(),[this](String::ConstSharedPtr m){auto o=QJsonDocument::fromJson(QByteArray::fromStdString(m->data)).object();QMetaObject::invokeMethod(this,[this,o](){radius_=o.value("radius").toDouble(.2);max_height_=o.value("max_height").toDouble(.35);all_=true;},Qt::QueuedConnection);});
  plane_=node->create_subscription<geometry_msgs::msg::Vector3Stamped>("/navigation/ground_plane",rclcpp::QoS(1).transient_local(),[this](geometry_msgs::msg::Vector3Stamped::ConstSharedPtr m){QMetaObject::invokeMethod(this,[this,m](){if(m->header.frame_id=="map"&&std::isfinite(m->vector.z)){a_=m->vector.x;b_=m->vector.y;c_=m->vector.z;}},Qt::QueuedConnection);});
  line_=std::make_unique<rviz_rendering::BillboardLine>(scene_manager_);line_->setMaxPointsPerLine(65);line_->setLineWidth(.025);line_->setColor(0.,1.,1.,1.);
  timer_.start();
 }
 void activate() override {setStatus("按住左键涂抹：圆圈内从上到下全部擦除。Ctrl+Z 撤销，面板按钮另存，Esc 退出。");}
 void deactivate() override {finish();if(line_)line_->clear();}
 void finish(){if(!stroke_.isEmpty()){pub_->publish(message({{"op","end"},{"stroke",stroke_}}));stroke_.clear();}}
 int processKeyEvent(QKeyEvent *e,rviz_common::RenderPanel*) override {
  if(e->key()==Qt::Key_Escape){finish();return Finished|Render;}
  if(e->modifiers()&Qt::ControlModifier){if(e->key()==Qt::Key_Z)pub_->publish(message({{"op","undo"}}));if(e->key()==Qt::Key_S)pub_->publish(message({{"op","save"}}));}
  return Render;
 }
 int processMouseEvent(rviz_common::ViewportMouseEvent &e) override {
  if(context_->getFixedFrame()!="map"){setStatus("请将 Fixed Frame 设为 map 后编辑");return Render;}
  auto cam=e.panel->getViewController()->getCamera();
  auto ray=cam->getCameraToViewportRay(float(e.x)/std::max(1,e.panel->width()),float(e.y)/std::max(1,e.panel->height()));
  // Project onto the near-ground editing band, not a roof or costmap graphic.
  Ogre::Plane plane(Ogre::Vector3(-a_,-b_,1.),Ogre::Vector3(0.,0.,c_+(all_?0.15:std::min(.15,max_height_*.5))));
  auto hit=ray.intersects(plane);if(!hit.first||hit.second<0){if(e.leftUp())finish();return Render;}
  auto p=ray.getPoint(hit.second);line_->clear();
  for(int i=0;i<=64;i++){double t=i*2.*M_PI/64;line_->addPoint(Ogre::Vector3(p.x+radius_*std::cos(t),p.y+radius_*std::sin(t),p.z+.015));}
  if(e.leftDown()){finish();stroke_=QUuid::createUuid().toString();}
  if((e.left()||e.leftDown()||e.leftUp())&&!stroke_.isEmpty()&&(timer_.elapsed()>45||e.leftDown()||e.leftUp())){
   pub_->publish(message({{"op","paint"},{"stroke",stroke_},{"x",double(p.x)},{"y",double(p.y)},{"radius",radius_},{"max_height",max_height_},{"all_heights",all_}}));timer_.restart();
  }
  if(e.leftUp())finish();return Render;
 }
private:
 rclcpp::Publisher<String>::SharedPtr pub_;rclcpp::Subscription<String>::SharedPtr settings_;
 rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr plane_;
 std::unique_ptr<rviz_rendering::BillboardLine> line_;QElapsedTimer timer_;QString stroke_;
 double radius_{.2},max_height_{.35},a_{0.},b_{0.},c_{-.27};bool all_{true};
};
class EditorPanel:public rviz_common::Panel {
public:
 EditorPanel(QWidget *parent=nullptr):rviz_common::Panel(parent){
  auto layout=new QVBoxLayout(this);auto help=new QLabel("选工具栏“点云橡皮擦”，按住左键涂抹。\n圆圈范围内从上到下全部擦除，不限高度。",this);help->setWordWrap(true);layout->addWidget(help);
  auto form=new QFormLayout;radius_=new QDoubleSpinBox(this);radius_->setRange(.05,1.5);radius_->setSingleStep(.05);radius_->setValue(.20);radius_->setSuffix(" m");form->addRow("笔刷半径",radius_);
  layout->addLayout(form);
  for(auto pair:{std::pair<const char*,const char*>("暂停导航","pause"),{"撤销上一笔  Ctrl+Z","undo"},{"另存清理后的 PCD","save"}}){auto button=new QPushButton(pair.first,this);layout->addWidget(button);std::string op=pair.second;connect(button,&QPushButton::clicked,this,[this,op](){if(pub_)pub_->publish(message({{"op",QString::fromStdString(op)}}));});}
  status_=new QLabel("等待编辑服务…",this);status_->setWordWrap(true);status_->setTextInteractionFlags(Qt::TextSelectableByMouse);layout->addWidget(status_);layout->addStretch();
  connect(radius_,QOverload<double>::of(&QDoubleSpinBox::valueChanged),this,[this](double){settings();});
 }
 void onInitialize() override {
  auto node=getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();pub_=node->create_publisher<String>("/map_editor/command",20);settings_pub_=node->create_publisher<String>("/map_editor/settings",rclcpp::QoS(1).transient_local());
  status_sub_=node->create_subscription<String>("/map_editor/status",rclcpp::QoS(1).transient_local(),[this](String::ConstSharedPtr m){QString text=QString::fromStdString(m->data);QMetaObject::invokeMethod(this,[this,text](){status_->setText(text);},Qt::QueuedConnection);});settings();
 }
 void settings(){if(settings_pub_)settings_pub_->publish(message({{"radius",radius_->value()},{"max_height",.35},{"all_heights",true}}));}
private:
 QDoubleSpinBox *radius_;QLabel *status_;
 rclcpp::Publisher<String>::SharedPtr pub_,settings_pub_;rclcpp::Subscription<String>::SharedPtr status_sub_;
};
}
PLUGINLIB_EXPORT_CLASS(rviz_cloud_editor::EraserTool,rviz_common::Tool)
PLUGINLIB_EXPORT_CLASS(rviz_cloud_editor::EditorPanel,rviz_common::Panel)
