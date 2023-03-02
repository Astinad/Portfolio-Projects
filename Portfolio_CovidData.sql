Select *
From [ PortfolioProject].dbo.CovidDeaths
Where continent is NOT NULL
Order By 3,4

Select *
FROM [ PortfolioProject].dbo.CovidVaccinations
Order By 3,4

Select location,date,total_cases,new_cases,total_deaths,population
From [ PortfolioProject].dbo.CovidDeaths
Order by 1,2

--Total Cases vs Total Deaths
-- Shows likelihood of dying if you contract covid in your country
Select location,date,total_cases,total_deaths,(total_deaths/total_cases)*100 AS DeathPercentage
From [ PortfolioProject].dbo.CovidDeaths
Where location like '%India%'
Order by 1,2

--Total Cases vs Population
-- Shows what percentage of population infected with Covid
Select location, date, population, (total_deaths/population)*100 AS DeathPercentage
From [ PortfolioProject].dbo.CovidDeaths
Where location like '%India%'
Order by 1,2

-- Countries with Highest Infection Rate compared to Population

Select location, population, MAX(total_cases) as HighestInfectionCount,  
Max((total_cases/population))*100 as PercentPopulationInfected
From [ PortfolioProject].dbo.CovidDeaths
--Where location like '%India%'
Group by location, population
order by PercentPopulationInfected desc

-- Countries with Highest Death Count per Population

Select Location, MAX(cast(Total_deaths as int)) as TotalDeathCount
From [ PortfolioProject].dbo.CovidDeaths
--Where location like '%India%'
Where continent is not null 
Group by location
order by TotalDeathCount desc

-- Continents with the highest death count per population
Select continent, MAX(cast(Total_deaths as int)) as TotalDeathCount
From [ PortfolioProject].dbo.CovidDeaths
Where continent is not null 
Group by continent
order by TotalDeathCount desc

-- GLOBAL NUMBERS
Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, 
SUM(cast(new_deaths as int))/SUM(New_Cases)*100 as DeathPercentage
From [ PortfolioProject].dbo.CovidDeaths
where continent is not null 
--Group By date
order by 1,2

-- Total Population vs Vaccinations
-- Shows Percentage of Population that has recieved at least one Covid Vaccine

Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) 
  OVER (Partition by dea.Location Order by dea.location, dea.Date) as PeopleVaccinated
From [ PortfolioProject].dbo.CovidDeaths dea
Join [ PortfolioProject].dbo.CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null 
order by 2,3

--CTE
With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, PeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as PeopleVaccinated
--, (PeopleVaccinated/population)*100
From [ PortfolioProject].dbo.CovidDeaths dea
Join [ PortfolioProject].dbo.CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null 
--order by 2,3
)
Select *, (PeopleVaccinated/Population)*100
From PopvsVac


--Temp Table
DROP Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
PeopleVaccinated numeric
)

Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) 
OVER (Partition by dea.Location Order by dea.location, dea.Date) as PeopleVaccinated
From [ PortfolioProject].dbo.CovidDeaths dea
Join [ PortfolioProject].dbo.CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date

Select *, (PeopleVaccinated/Population)*100
From #PercentPopulationVaccinated

-- Creating View to store data for later visualizations
Create View PercentPopulationVaccinated as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) 
OVER (Partition by dea.Location Order by dea.location, dea.Date) as PeopleVaccinated
From [ PortfolioProject].dbo.CovidDeaths dea
Join [ PortfolioProject].dbo.CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null 

Select *
From PercentPopulationVaccinated
