from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()


urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'demo.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'demo.views.default.default'),
    url(r'^testpage/', 'demo.views.testpage.predict'),
    (r'^site_media/(?P<path>.*)', 'django.views.static.serve',{'document_root': '/home/daizhen/projects/ImagesCategory/data/gray_images'}),
)
